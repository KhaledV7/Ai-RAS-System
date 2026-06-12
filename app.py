"""
RAS SENTINEL - app.py  (STEP 3: The Live Dashboard)
===================================================
The screen the judges watch. It wraps the EXACT agent loop already verified in
run_pipeline.py and gives it a face:

    MONITOR -> SUSPECT -> INTERROGATE -> VERDICT -> (SUPPRESS / ESCALATE) -> LOG

The memorable moment: on the same "low reading" symptom, the verdict banner flips
GREEN (clear) / AMBER (hardware fault, alarm suppressed) / RED (real crisis) based on
the physical thermal evidence -- not on a naive threshold.

Run it locally:
    pip install streamlit altair pandas numpy scikit-learn
    streamlit run app.py
(keep this file next to ras_sentinel_engine.py and the two scenario CSVs)

Honesty notes kept visible in the UI:
  * The thermal layer is SIMULATION / DIGITAL-TWIN mode (field calibration pending).
  * is_injected is used ONLY to pace the playback (skip calm baseline). The agent never
    sees it -- it decides purely from the readings, exactly as in run_pipeline.py.
"""

import time
import sqlite3

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import ras_sentinel_engine as eng

# --------------------------------------------------------------------------- #
# Loop constants -- identical to run_pipeline.py so behaviour matches exactly
# --------------------------------------------------------------------------- #
WINDOW = 10          # trailing rows needed to compute slope features
COOLDOWN = 250       # readings to wait after a verdict before interrogating again

SCENARIOS = {
    "Sensor fouling  (false alarm expected)": "sensor_baseline_fail.csv",
    "Pool crisis  (real crisis expected)":    "pool_crisis.csv",
}

# Palette -- "deep tank / control room", grounded in the subject (water + oxygen)
C_BG      = "#081019"
C_PANEL   = "#0e1d28"
C_LINE    = "#1d3645"
C_INK     = "#eaf4f4"
C_MUTED   = "#7d97a3"
C_OXY     = "#2bd4c0"   # oxygen cyan  -> DO / healthy / clean
C_VIOLET  = "#9b8cff"   # pH series
C_AMBER   = "#f5b13d"   # hardware fault / maintenance
C_CRISIS  = "#ff5d62"   # real crisis


# --------------------------------------------------------------------------- #
# The agent, as a generator (pure logic, no UI). Mirrors run_pipeline exactly.
# --------------------------------------------------------------------------- #
def stream_agent(df: pd.DataFrame, clf, observer: "eng.Observer"):
    """Yield (i, row, state, event) for every reading.

    state  : one of MONITOR / SUSPECT / INTERROGATE / VERDICT / COOLDOWN
    event  : None, or a dict describing a completed interrogation + verdict.
    """
    cooldown = 0
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        state, event = "MONITOR", None

        if cooldown > 0:                       # still acting on the last verdict
            cooldown -= 1
            state = "COOLDOWN"
            yield i, row, state, event
            continue

        window = df.iloc[max(0, i - WINDOW + 1): i + 1]
        is_anom = observer.score_row(window)
        if is_anom:
            state = "SUSPECT"
        escalate = observer.update_debounce(is_anom)

        if escalate:
            # ---- INTERROGATE: fire the thermal pulse on THIS event ----
            k = float(row.get("k_value", 0.85))     # swap point for real sensor data
            curve = eng.get_thermal_curve(k)
            thermal_verdict, conf, k_est = eng.classify_pulse(clf, curve)

            # ---- VERDICT ----
            if "HARDWARE FAULT" in thermal_verdict:
                final = "HARDWARE FAULT (biofouling)"
                action = "Alarm suppressed - maintenance ticket: clean the DO probe"
                severity = "suppress"
                detail = "Probe insulated by biofilm: heat is trapped (low k). The water is fine."
            else:
                viol, text, _ = eng.diagnose_water(row)
                if viol:
                    final = "REAL WATER CRISIS"
                    action = "Escalate - activate emergency aeration / intervention"
                    severity = "crisis"
                    detail = text
                else:
                    final = "Anomaly within SAMAQ limits"
                    action = "Probe clean, water still safe - keep monitoring"
                    severity = "clear"
                    detail = "Sensor verified clean and every parameter is inside the SAMAQ limits."

            eng.log_decision(row, k_est, final, conf)
            event = {
                "curve": curve, "k_est": k_est, "conf": conf,
                "final": final, "action": action, "severity": severity,
                "detail": detail, "k_input": k,
            }
            observer._streak = 0
            cooldown = COOLDOWN
            state = "VERDICT"

        yield i, row, state, event


# --------------------------------------------------------------------------- #
# Small UI builders
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

        .stApp {{ background: radial-gradient(1200px 600px at 70% -10%, #0c2230 0%, {C_BG} 55%); }}
        .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {C_INK}; }}
        h1, h2, h3 {{ font-family: 'Sora', sans-serif; letter-spacing: -0.01em; }}

        /* header */
        .hdr {{ display:flex; align-items:center; justify-content:space-between;
               border-bottom:1px solid {C_LINE}; padding-bottom:.7rem; margin-bottom:1.1rem; }}
        .hdr .title {{ font-family:'Sora'; font-weight:700; font-size:1.55rem; }}
        .hdr .title small {{ color:{C_OXY}; font-weight:600; }}
        .hdr .sub {{ color:{C_MUTED}; font-size:.82rem; font-family:'JetBrains Mono'; }}
        .simbadge {{ font-family:'JetBrains Mono'; font-size:.72rem; color:{C_AMBER};
                    border:1px solid {C_AMBER}55; border-radius:999px; padding:.28rem .7rem;
                    background:{C_AMBER}12; white-space:nowrap; }}

        /* state machine chips */
        .chips {{ display:flex; gap:.4rem; flex-wrap:wrap; margin:.2rem 0 1rem; }}
        .chip {{ font-family:'JetBrains Mono'; font-size:.72rem; letter-spacing:.06em;
                color:{C_MUTED}; border:1px solid {C_LINE}; border-radius:6px;
                padding:.3rem .6rem; background:{C_PANEL}; transition:all .15s; }}
        .chip.on {{ color:{C_BG}; background:{C_OXY}; border-color:{C_OXY}; font-weight:600; }}
        .chip.on.amber  {{ background:{C_AMBER}; border-color:{C_AMBER}; }}
        .chip.on.crisis {{ background:{C_CRISIS}; border-color:{C_CRISIS}; color:{C_INK}; }}

        /* verdict banner */
        .banner {{ border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1.1rem;
                  border:1px solid {C_LINE}; background:{C_PANEL};
                  display:flex; align-items:center; gap:1rem; }}
        .banner .dot {{ width:14px; height:14px; border-radius:50%; flex:none;
                       box-shadow:0 0 0 6px #ffffff10; }}
        .banner .v {{ font-family:'Sora'; font-weight:700; font-size:1.3rem; line-height:1.1; }}
        .banner .a {{ color:{C_MUTED}; font-size:.9rem; margin-top:.15rem; }}
        .banner.clear   {{ border-color:{C_OXY}55;   background:linear-gradient(90deg,{C_OXY}14,transparent); }}
        .banner.clear   .dot {{ background:{C_OXY}; }}
        .banner.suppress{{ border-color:{C_AMBER}66; background:linear-gradient(90deg,{C_AMBER}16,transparent); }}
        .banner.suppress .dot {{ background:{C_AMBER}; }}
        .banner.crisis  {{ border-color:{C_CRISIS}77; background:linear-gradient(90deg,{C_CRISIS}1c,transparent); }}
        .banner.crisis  .dot {{ background:{C_CRISIS}; animation:pulse 1s infinite; }}
        @keyframes pulse {{ 0%{{box-shadow:0 0 0 0 {C_CRISIS}66;}} 100%{{box-shadow:0 0 0 12px {C_CRISIS}00;}} }}

        /* metric cards */
        .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin-bottom:1.1rem; }}
        .card {{ background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:12px; padding:.8rem .9rem; }}
        .card .lab {{ color:{C_MUTED}; font-size:.72rem; font-family:'JetBrains Mono'; letter-spacing:.05em; }}
        .card .val {{ font-family:'JetBrains Mono'; font-weight:600; font-size:1.55rem; margin-top:.15rem; }}
        .card .lim {{ font-size:.7rem; color:{C_MUTED}; margin-top:.1rem; }}
        .card.ok   {{ }}
        .card.bad  {{ border-color:{C_CRISIS}88; }}
        .card.bad .val {{ color:{C_CRISIS}; }}

        /* panels */
        .panel {{ background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:12px; padding:1rem 1.1rem; }}
        .panel h4 {{ font-family:'Sora'; font-size:.95rem; margin:0 0 .2rem; }}
        .panel .note {{ color:{C_MUTED}; font-size:.74rem; font-family:'JetBrains Mono'; }}
        .kbig {{ font-family:'JetBrains Mono'; font-size:2rem; font-weight:600; }}

        section[data-testid="stSidebar"] {{ background:{C_PANEL}; border-right:1px solid {C_LINE}; }}
        .stButton>button {{ font-family:'Sora'; font-weight:600; border-radius:10px;
                           border:1px solid {C_OXY}; background:{C_OXY}; color:{C_BG}; }}
        .stButton>button:hover {{ background:{C_OXY}dd; color:{C_BG}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


STATES = ["MONITOR", "SUSPECT", "INTERROGATE", "VERDICT"]


def chips_html(active: str, severity: str | None) -> str:
    cls = {"suppress": "amber", "crisis": "crisis"}.get(severity or "", "")
    shown = "INTERROGATE" if active == "VERDICT" else active
    out = ['<div class="chips">']
    for s in STATES:
        on = "on" if s == shown or (active == "VERDICT" and s in ("INTERROGATE", "VERDICT")) else ""
        extra = cls if (on and s == "VERDICT") else ""
        out.append(f'<div class="chip {on} {extra}">{s}</div>')
    out.append("</div>")
    return "".join(out)


def cards_html(row: dict) -> str:
    do, ph, nh4, temp = row["DO"], row["pH"], row["NH4_N"], row["Temp"]
    do_bad  = do  < eng.SAMAQ["DO_MIN"]
    ph_bad  = ph  < eng.SAMAQ["PH_MIN"] or ph > eng.SAMAQ["PH_MAX"]
    nh4_bad = nh4 > eng.SAMAQ["NH4_MAX"]
    tp_bad  = temp < eng.TEMP_COMFORT["TEMP_MIN"] or temp > eng.TEMP_COMFORT["TEMP_MAX"]

    def card(lab, val, unit, lim, bad):
        return (f'<div class="card {"bad" if bad else "ok"}">'
                f'<div class="lab">{lab}</div>'
                f'<div class="val">{val:.2f}<span style="font-size:.8rem;color:{C_MUTED}"> {unit}</span></div>'
                f'<div class="lim">{lim}</div></div>')

    return ('<div class="cards">'
            + card("DISSOLVED O\u2082", do, "mg/L", f"SAMAQ min {eng.SAMAQ['DO_MIN']}", do_bad)
            + card("pH", ph, "", f"SAMAQ {eng.SAMAQ['PH_MIN']}\u2013{eng.SAMAQ['PH_MAX']}", ph_bad)
            + card("AMMONIA (NH\u2084)", nh4, "mg/L", f"SAMAQ max {eng.SAMAQ['NH4_MAX']}", nh4_bad)
            + card("TEMP", temp, "\u00b0C", "tilapia band 22\u201332", tp_bad)
            + "</div>")


def banner_html(event: dict | None) -> str:
    if event is None:
        return ('<div class="banner clear"><div class="dot"></div><div>'
                f'<div class="v">Monitoring \u2014 water trusted</div>'
                f'<div class="a">No persistent anomaly. The agent is watching rate-of-change, not just raw values.</div>'
                "</div></div>")
    sev = event["severity"]
    cls = {"clear": "clear", "suppress": "suppress", "crisis": "crisis"}[sev]
    return (f'<div class="banner {cls}"><div class="dot"></div><div>'
            f'<div class="v">{event["final"]}</div>'
            f'<div class="a">{event["action"]} \u00b7 {event["detail"]}</div>'
            "</div></div>")


def param_chart(hist: pd.DataFrame):
    """Recent trace of the three parameters that move, with the SAMAQ limit lines."""
    long = hist.melt("t", value_vars=["DO", "pH", "NH4_N"], var_name="param", value_name="val")
    color = alt.Color("param:N",
                      scale=alt.Scale(domain=["DO", "pH", "NH4_N"],
                                      range=[C_OXY, C_VIOLET, C_AMBER]),
                      legend=alt.Legend(orient="top", title=None))
    line = (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("t:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("val:Q", title="mg/L  /  pH",
                            scale=alt.Scale(domain=[0, 11])),
                    color=color))
    rules = pd.DataFrame({"y": [eng.SAMAQ["DO_MIN"], eng.SAMAQ["PH_MAX"]],
                          "lab": ["SAMAQ 5.0  (DO floor / NH\u2084 ceiling)", "pH ceiling 9.5"]})
    rule = (alt.Chart(rules).mark_rule(color=C_CRISIS, strokeDash=[4, 4], opacity=.7)
            .encode(y="y:Q"))
    return (line + rule).properties(height=240, background="transparent").configure_view(
        strokeWidth=0).configure_axis(grid=True, gridColor=C_LINE, gridOpacity=.4,
                                      labelColor=C_MUTED, titleColor=C_MUTED)


def thermal_chart(curve: np.ndarray, severity: str):
    col = {"suppress": C_AMBER, "crisis": C_OXY, "clear": C_OXY}[severity]
    t = np.linspace(0, 3, len(curve))
    d = pd.DataFrame({"t": t, "temp": curve})
    area = (alt.Chart(d).mark_area(opacity=.18, color=col)
            .encode(x=alt.X("t:Q", title="seconds"), y=alt.Y("temp:Q", title="\u0394T \u00b0C")))
    line = (alt.Chart(d).mark_line(strokeWidth=2.5, color=col)
            .encode(x="t:Q", y="temp:Q"))
    return (area + line).properties(height=200, background="transparent").configure_view(
        strokeWidth=0).configure_axis(grid=True, gridColor=C_LINE, gridOpacity=.4,
                                      labelColor=C_MUTED, titleColor=C_MUTED)


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="RAS Sentinel", page_icon="\U0001F41F", layout="wide")
    inject_css()

    st.markdown(
        '<div class="hdr">'
        '<div><div class="title">RAS&nbsp;<small>SENTINEL</small></div>'
        '<div class="sub">edge AI gateway \u00b7 verify before you alarm</div></div>'
        '<div class="simbadge">\u25CF THERMAL LAYER: SIMULATION / DIGITAL-TWIN \u2014 field calibration pending</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- sidebar controls ----
    with st.sidebar:
        st.markdown("### Scenario")
        scenario_label = st.radio("scenario", list(SCENARIOS.keys()), label_visibility="collapsed")
        path = SCENARIOS[scenario_label]
        speed = st.slider("Playback speed (sec / frame)", 0.0, 0.25, 0.05, 0.01)
        ff_calm = st.checkbox("Fast-forward calm baseline", value=True)
        clear_log = st.checkbox("Clear log before run", value=True)
        run = st.button("\u25B6  Run scenario", use_container_width=True)
        st.markdown(
            f'<p class="note" style="margin-top:1rem">The agent never sees the scenario label '
            f'or the <code>is_injected</code> tag \u2014 it decides only from the readings.</p>',
            unsafe_allow_html=True)

    # ---- placeholders (updated in place during playback) ----
    chips_ph   = st.empty()
    banner_ph  = st.empty()
    cards_ph   = st.empty()
    col_l, col_r = st.columns([1.35, 1])
    with col_l:
        st.markdown("##### Live water parameters")
        chart_ph = st.empty()
    with col_r:
        st.markdown("##### Thermal interrogation")
        thermal_head_ph = st.empty()
        thermal_chart_ph = st.empty()
    st.markdown("##### Offline decision log  ·  edge_telemetry.db")
    log_ph = st.empty()

    # ---- initial idle render ----
    chips_ph.markdown(chips_html("MONITOR", None), unsafe_allow_html=True)
    banner_ph.markdown(banner_html(None), unsafe_allow_html=True)
    thermal_head_ph.markdown(
        '<div class="panel"><h4>Standing by</h4>'
        '<p class="note">No interrogation yet. When the Observer suspects a problem, the agent '
        'pulses the probe\u2019s thermistor and reads the cooling curve here.</p></div>',
        unsafe_allow_html=True)

    if not run:
        try:
            df0 = pd.read_csv(path)
            cards_ph.markdown(cards_html(df0.iloc[0].to_dict()), unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"Could not find {path}. Run generate_datasets.py first, and keep this "
                     f"file next to the CSVs and ras_sentinel_engine.py.")
        st.info("Pick a scenario on the left and press **Run scenario**.")
        return

    # ---- load + prepare ----
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Could not find {path}. Run generate_datasets.py first.")
        return

    eng.init_db()
    if clear_log:
        conn = sqlite3.connect(eng.DB_PATH)
        conn.execute("DELETE FROM verified_logs")
        conn.commit(); conn.close()

    clf = eng.bootstrap_thermal_classifier()
    obs = eng.Observer()
    obs.train(df.iloc[:eng.BASELINE_ROWS])

    # pacing: skip through the calm baseline, slow down in the fault region
    if "is_injected" in df.columns and (df["is_injected"] == 1).any():
        fault_start = int(df.index[df["is_injected"] == 1][0])
    else:
        fault_start = len(df)
    calm_every = max(1, fault_start // 12)                       # ~12 calm frames
    live_every = max(1, (len(df) - fault_start) // 160)          # ~160 live frames

    hist_rows = []
    last_event = None
    last_sev = None
    progress = st.sidebar.progress(0.0)

    for i, row, state, event in stream_agent(df, clf, obs):
        hist_rows.append({"t": i, "DO": row["DO"], "pH": row["pH"], "NH4_N": row["NH4_N"]})
        if event is not None:
            last_event, last_sev = event, event["severity"]

        in_calm = i < fault_start
        render = (event is not None) or (i == len(df) - 1) \
            or (i % (calm_every if (in_calm and ff_calm) else live_every) == 0)
        if not render:
            continue

        # update the four live regions
        chips_ph.markdown(chips_html(state, last_sev), unsafe_allow_html=True)
        cards_ph.markdown(cards_html(row), unsafe_allow_html=True)
        if last_event is not None:
            banner_ph.markdown(banner_html(last_event), unsafe_allow_html=True)

        hist = pd.DataFrame(hist_rows).tail(240)
        chart_ph.altair_chart(param_chart(hist), use_container_width=True)

        if event is not None:
            thermal_head_ph.markdown(
                f'<div class="panel"><h4>Probe interrogated</h4>'
                f'<div class="kbig" style="color:{C_OXY if event["severity"]!="suppress" else C_AMBER}">'
                f'k = {event["k_est"]:.2f}</div>'
                f'<p class="note">cooling constant \u00b7 confidence {event["conf"]:.0%} '
                f'\u00b7 SIMULATION MODE</p></div>',
                unsafe_allow_html=True)
            thermal_chart_ph.altair_chart(thermal_chart(event["curve"], event["severity"]),
                                          use_container_width=True)
            log_ph.dataframe(eng.fetch_logs(8), use_container_width=True, hide_index=True)
            time.sleep(max(speed, 0.05) + 0.9)     # let judges read the verdict
        else:
            time.sleep(speed)

        progress.progress(min(1.0, (i + 1) / len(df)))

    progress.progress(1.0)
    # final log table
    log_ph.dataframe(eng.fetch_logs(12), use_container_width=True, hide_index=True)
    st.success("Scenario complete \u2014 every decision above was logged offline first, "
               "then would sync to the cloud. That is the edge-to-cloud story.")


if __name__ == "__main__":
    main()
