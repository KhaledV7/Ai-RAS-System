"""
RAS SENTINEL - app.py  (Operations Console)
===========================================
A production-style edge console for a single RAS tank, not a scenario tester.

Two tabs:
  * LIVE MONITOR     - one continuous farm feed (1 reading = 1 minute, on a farm clock).
                       The agent runs MONITOR -> SUSPECT -> INTERROGATE -> VERDICT live.
  * MANUAL TEST      - type ANY water values + pick the probe condition, and the agent
                       diagnoses + interrogates them on the spot. This is the "what-if".

Honest design notes kept visible:
  * The thermal layer is SIMULATION / DIGITAL-TWIN (field calibration pending).
  * In the real product the thermal pulse MEASURES the probe (k). Offline there is no
    probe, so the Manual tab lets you state the probe condition (clean / fouled); when a
    real sonde is wired in, that control disappears and the pulse fills in k by itself.

Run locally:
    pip install streamlit altair pandas numpy scikit-learn
    streamlit run app.py
Keep this file next to ras_sentinel_engine.py.
"""

import time
import sqlite3
import datetime as dt

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import ras_sentinel_engine as eng

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
WINDOW = 10            # trailing rows for slope features (matches the engine)
COOLDOWN_LIVE = 14     # readings to wait after a verdict before re-interrogating
TRAIN_ROWS = 100       # calm rows the Observer learns "normal" from
FARM = "Qassim RAS Facility"
TANK = "Tank A-03  ·  Nile tilapia"
LIVE_PACE = 0.8        # seconds per reading in the live monitor (steady real-time cadence)

# Palette - "deep tank / control room"
C_BG     = "#081019"
C_PANEL  = "#0e1d28"
C_LINE   = "#1d3645"
C_INK    = "#eaf4f4"
C_MUTED  = "#7d97a3"
C_OXY    = "#2bd4c0"   # oxygen cyan -> healthy / clean
C_VIOLET = "#9b8cff"   # pH series
C_AMBER  = "#f5b13d"   # hardware fault / maintenance
C_CRISIS = "#ff5d62"   # real crisis


# --------------------------------------------------------------------------- #
# The farm feed: ONE continuous day, no scenario labels.
# calm -> a probe slowly fouls (DO looks bad, water fine) -> probe cleaned ->
# a genuine water crisis (ammonia + pH climb, probe clean) -> calm tail.
# --------------------------------------------------------------------------- #
def build_farm_feed(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(2025, 6, 24, 6, 0)
    rows = []

    def push(do, ph, nh4, k):
        i = len(rows)
        rows.append({
            "clock": t0 + dt.timedelta(minutes=i),
            "DO": round(float(do), 2), "pH": round(float(ph), 2),
            "NH4_N": round(float(nh4), 2), "Temp": round(26.8 + rng.normal(0, 0.06), 2),
            "k_value": k,
        })

    def calm(n, k=0.85):
        for _ in range(n):
            push(6.10 + rng.normal(0, 0.06), 7.69 + rng.normal(0, 0.03),
                 1.00 + rng.normal(0, 0.05), k)

    calm(110)                                            # 1. normal morning
    n = 45                                               # 2. biofouling: DO sags, probe FOULED
    for j in range(n):
        push(6.10 - (6.10 - 3.30) * (j + 1) / n + rng.normal(0, 0.05),
             7.69 + rng.normal(0, 0.03), 1.00 + rng.normal(0, 0.05), 0.12)
    calm(28)                                             # 3. probe cleaned -> back to normal
    n = 45                                               # 4. real crisis: NH4 + pH climb, probe CLEAN
    for j in range(n):
        push(6.05 + rng.normal(0, 0.06),
             7.69 + (9.80 - 7.69) * (j + 1) / n + rng.normal(0, 0.03),
             1.00 + (9.00 - 1.00) * (j + 1) / n + rng.normal(0, 0.10), 0.85)
    calm(22)                                             # 5. resolved tail
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shared decision logic (used by BOTH tabs) - mirrors run_pipeline exactly
# --------------------------------------------------------------------------- #
def evaluate(row: dict, k: float, clf) -> dict:
    """Fire the thermal pulse on this reading and return the agent's verdict."""
    curve = eng.get_thermal_curve(k)
    thermal_verdict, conf, k_est = eng.classify_pulse(clf, curve)
    viol, text, _ = eng.diagnose_water(row)

    if "HARDWARE FAULT" in thermal_verdict:
        final = "HARDWARE FAULT (biofouling)"
        action = "Alarm suppressed - maintenance ticket raised: clean the probe"
        sev = "suppress"
        detail = "Probe insulated by biofilm - the heat is trapped (low k). The water itself is fine."
    elif viol:
        final = "REAL WATER CRISIS"
        action = "Escalate - activate emergency aeration / intervention"
        sev = "crisis"
        detail = text
    else:
        final = "Verified clean - water safe"
        action = "Probe clean and every parameter within SAMAQ limits - keep monitoring"
        sev = "clear"
        detail = "Sensor verified clean and all parameters inside the SAMAQ limits."

    return {"curve": curve, "k_est": k_est, "conf": conf, "final": final,
            "action": action, "severity": sev, "detail": detail}


# --------------------------------------------------------------------------- #
# Cached models + feed
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_models():
    eng.init_db()
    feed = build_farm_feed()
    obs = eng.Observer()
    obs.train(feed.iloc[:TRAIN_ROWS])
    clf = eng.bootstrap_thermal_classifier()
    return feed, obs, clf


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
    .stApp {{ background: radial-gradient(1200px 600px at 72% -12%, #0c2230 0%, {C_BG} 55%); }}
    header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    .block-container {{ padding-top: 2.6rem; max-width: 1180px; }}
    html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{C_INK}; }}
    h1,h2,h3,h4 {{ font-family:'Sora',sans-serif; letter-spacing:-.01em; }}

    .hdr {{ display:flex; align-items:center; justify-content:space-between;
            border-bottom:1px solid {C_LINE}; padding-bottom:.7rem; margin-bottom:.4rem; }}
    .hdr .title {{ font-family:'Sora'; font-weight:700; font-size:1.5rem; }}
    .hdr .title small {{ color:{C_OXY}; font-weight:600; }}
    .hdr .sub {{ color:{C_MUTED}; font-size:.8rem; font-family:'JetBrains Mono'; }}
    .simbadge {{ font-family:'JetBrains Mono'; font-size:.7rem; color:{C_AMBER};
                 border:1px solid {C_AMBER}55; border-radius:999px; padding:.26rem .7rem;
                 background:{C_AMBER}12; white-space:nowrap; }}

    .statusbar {{ display:flex; gap:1.4rem; align-items:center; color:{C_MUTED};
                  font-family:'JetBrains Mono'; font-size:.78rem; margin:.5rem 0 1rem; }}
    .statusbar b {{ color:{C_INK}; }}
    .clock {{ color:{C_OXY}; font-weight:600; }}

    .chips {{ display:flex; gap:.4rem; flex-wrap:wrap; margin:.1rem 0 1rem; }}
    .chip {{ font-family:'JetBrains Mono'; font-size:.7rem; letter-spacing:.06em; color:{C_MUTED};
             border:1px solid {C_LINE}; border-radius:6px; padding:.3rem .6rem; background:{C_PANEL}; }}
    .chip.on {{ color:{C_BG}; background:{C_OXY}; border-color:{C_OXY}; font-weight:600; }}
    .chip.on.amber  {{ background:{C_AMBER}; border-color:{C_AMBER}; }}
    .chip.on.crisis {{ background:{C_CRISIS}; border-color:{C_CRISIS}; color:{C_INK}; }}

    .banner {{ border-radius:14px; padding:1.15rem 1.35rem; margin-bottom:1.05rem;
               border:1px solid {C_LINE}; background:{C_PANEL}; display:flex; align-items:center; gap:1rem; }}
    .banner .dot {{ width:14px; height:14px; border-radius:50%; flex:none; box-shadow:0 0 0 6px #ffffff10; }}
    .banner .v {{ font-family:'Sora'; font-weight:700; font-size:1.25rem; line-height:1.1; }}
    .banner .a {{ color:{C_MUTED}; font-size:.88rem; margin-top:.15rem; }}
    .banner.clear    {{ border-color:{C_OXY}55;   background:linear-gradient(90deg,{C_OXY}14,transparent); }}
    .banner.clear .dot {{ background:{C_OXY}; }}
    .banner.suppress {{ border-color:{C_AMBER}66; background:linear-gradient(90deg,{C_AMBER}16,transparent); }}
    .banner.suppress .dot {{ background:{C_AMBER}; }}
    .banner.crisis   {{ border-color:{C_CRISIS}77; background:linear-gradient(90deg,{C_CRISIS}1c,transparent); }}
    .banner.crisis .dot {{ background:{C_CRISIS}; animation:pulse 1s infinite; }}
    .banner.busy {{ border-color:{C_VIOLET}66; background:linear-gradient(90deg,{C_VIOLET}16,transparent); }}
    .banner.busy .dot {{ background:{C_VIOLET}; animation:pulse 1s infinite; }}
    @keyframes pulse {{ 0%{{box-shadow:0 0 0 0 {C_CRISIS}66;}} 100%{{box-shadow:0 0 0 12px {C_CRISIS}00;}} }}

    .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin-bottom:1.05rem; }}
    .card {{ background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:12px; padding:.8rem .9rem; }}
    .card .lab {{ color:{C_MUTED}; font-size:.7rem; font-family:'JetBrains Mono'; letter-spacing:.05em; }}
    .card .val {{ font-family:'JetBrains Mono'; font-weight:600; font-size:1.5rem; margin-top:.12rem; }}
    .card .lim {{ font-size:.68rem; color:{C_MUTED}; margin-top:.1rem; }}
    .card.bad {{ border-color:{C_CRISIS}88; }}
    .card.bad .val {{ color:{C_CRISIS}; }}

    .panel {{ background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:12px; padding:1rem 1.1rem; }}
    .panel h4 {{ font-family:'Sora'; font-size:.92rem; margin:0 0 .25rem; }}
    .panel .note {{ color:{C_MUTED}; font-size:.73rem; font-family:'JetBrains Mono'; }}
    .kbig {{ font-family:'JetBrains Mono'; font-size:2rem; font-weight:600; }}

    section[data-testid="stSidebar"] {{ background:{C_PANEL}; border-right:1px solid {C_LINE}; }}
    .stButton>button {{ font-family:'Sora'; font-weight:600; border-radius:10px;
                        border:1px solid {C_OXY}; background:{C_OXY}; color:{C_BG}; }}
    .stButton>button:hover {{ background:{C_OXY}dd; color:{C_BG}; }}
    </style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
STATES = ["MONITOR", "SUSPECT", "INTERROGATE", "VERDICT"]


def chips_html(active: str, severity=None) -> str:
    cls = {"suppress": "amber", "crisis": "crisis"}.get(severity or "", "")
    out = ['<div class="chips">']
    for s in STATES:
        on = "on" if (s == active or (active == "VERDICT" and s in ("INTERROGATE", "VERDICT"))) else ""
        extra = cls if (on and s == "VERDICT") else ""
        out.append(f'<div class="chip {on} {extra}">{s}</div>')
    out.append("</div>")
    return "".join(out)


def cards_html(row: dict) -> str:
    do, ph, nh4, temp = row["DO"], row["pH"], row["NH4_N"], row["Temp"]
    bad = {
        "do":  do < eng.SAMAQ["DO_MIN"],
        "ph":  ph < eng.SAMAQ["PH_MIN"] or ph > eng.SAMAQ["PH_MAX"],
        "nh4": nh4 > eng.SAMAQ["NH4_MAX"],
        "tp":  temp < eng.TEMP_COMFORT["TEMP_MIN"] or temp > eng.TEMP_COMFORT["TEMP_MAX"],
    }

    def card(lab, val, unit, lim, b):
        return (f'<div class="card {"bad" if b else ""}"><div class="lab">{lab}</div>'
                f'<div class="val">{val:.2f}<span style="font-size:.78rem;color:{C_MUTED}"> {unit}</span></div>'
                f'<div class="lim">{lim}</div></div>')

    return ('<div class="cards">'
            + card("DISSOLVED O\u2082", do, "mg/L", f"SAMAQ min {eng.SAMAQ['DO_MIN']}", bad["do"])
            + card("pH", ph, "", f"SAMAQ {eng.SAMAQ['PH_MIN']}\u2013{eng.SAMAQ['PH_MAX']}", bad["ph"])
            + card("AMMONIA (NH\u2084)", nh4, "mg/L", f"SAMAQ max {eng.SAMAQ['NH4_MAX']}", bad["nh4"])
            + card("TEMP", temp, "\u00b0C", "tilapia 22\u201332", bad["tp"])
            + "</div>")


def banner_html(event, busy=False) -> str:
    if busy:
        return ('<div class="banner busy"><div class="dot"></div><div>'
                '<div class="v">Interrogating probe\u2026</div>'
                '<div class="a">Firing thermal pulse and reading the 30-second cooling curve.</div>'
                "</div></div>")
    if event is None:
        return ('<div class="banner clear"><div class="dot"></div><div>'
                '<div class="v">Monitoring \u2014 water trusted</div>'
                '<div class="a">No persistent anomaly. The agent watches rate-of-change, not just raw values.</div>'
                "</div></div>")
    cls = event["severity"]
    return (f'<div class="banner {cls}"><div class="dot"></div><div>'
            f'<div class="v">{event["final"]}</div>'
            f'<div class="a">{event["action"]} \u00b7 {event["detail"]}</div>'
            "</div></div>")


def thermal_idle_html() -> str:
    return ('<div class="panel"><h4>Probe idle</h4>'
            '<p class="note">No pulse running. The cooling constant k is only measured during an '
            'interrogation \u2014 it stays blank while the water is trusted.</p></div>')


def thermal_busy_html() -> str:
    return (f'<div class="panel"><h4>Pulsing\u2026</h4>'
            f'<div class="kbig" style="color:{C_VIOLET}">k = \u2014</div>'
            f'<p class="note">heating thermistor +5\u00b0C \u00b7 sampling 10 Hz \u00b7 SIMULATION MODE</p></div>')


def thermal_result_html(event, when="") -> str:
    col = C_AMBER if event["severity"] == "suppress" else C_OXY
    stamp = f' \u00b7 {when}' if when else ""
    return (f'<div class="panel"><h4>Probe interrogated{stamp}</h4>'
            f'<div class="kbig" style="color:{col}">k = {event["k_est"]:.2f}</div>'
            f'<p class="note">cooling constant \u00b7 confidence {event["conf"]:.0%} \u00b7 SIMULATION MODE</p></div>')


def param_chart(hist: pd.DataFrame):
    long = hist.melt("t", value_vars=["DO", "pH", "NH4_N"], var_name="param", value_name="val")
    color = alt.Color("param:N",
                      scale=alt.Scale(domain=["DO", "pH", "NH4_N"], range=[C_OXY, C_VIOLET, C_AMBER]),
                      legend=alt.Legend(orient="top", title=None))
    line = (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("t:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("val:Q", title="mg/L  /  pH", scale=alt.Scale(domain=[0, 11])),
                    color=color))
    rules = pd.DataFrame({"y": [eng.SAMAQ["DO_MIN"], eng.SAMAQ["PH_MAX"]]})
    rule = alt.Chart(rules).mark_rule(color=C_CRISIS, strokeDash=[4, 4], opacity=.7).encode(y="y:Q")
    return (line + rule).properties(height=240, background="transparent").configure_view(
        strokeWidth=0).configure_axis(grid=True, gridColor=C_LINE, gridOpacity=.4,
                                      labelColor=C_MUTED, titleColor=C_MUTED)


def thermal_chart(curve: np.ndarray, severity: str):
    col = C_AMBER if severity == "suppress" else C_OXY
    t = np.linspace(0, 30, len(curve))
    d = pd.DataFrame({"t": t, "temp": curve})
    area = alt.Chart(d).mark_area(opacity=.18, color=col).encode(
        x=alt.X("t:Q", title="seconds"), y=alt.Y("temp:Q", title="\u0394T \u00b0C"))
    line = alt.Chart(d).mark_line(strokeWidth=2.5, color=col).encode(x="t:Q", y="temp:Q")
    return (area + line).properties(height=200, background="transparent").configure_view(
        strokeWidth=0).configure_axis(grid=True, gridColor=C_LINE, gridOpacity=.4,
                                      labelColor=C_MUTED, titleColor=C_MUTED)


# --------------------------------------------------------------------------- #
# Shared streaming agent loop (used by the live monitor and the file runner)
# --------------------------------------------------------------------------- #
def build_console():
    P = {}
    P["clock"]  = st.empty()
    P["chips"]  = st.empty()
    P["banner"] = st.empty()
    P["cards"]  = st.empty()
    colL, colR = st.columns([1.35, 1])
    with colL:
        st.markdown("##### Live water parameters")
        P["chart"] = st.empty()
    with colR:
        st.markdown("##### Thermal interrogation")
        P["th_head"]  = st.empty()
        P["th_chart"] = st.empty()
    st.markdown("##### Offline decision log")
    P["log"] = st.empty()
    return P


def _clock_label(row, i):
    c = row.get("clock")
    if c is not None and hasattr(c, "strftime"):
        return c.strftime("%H:%M")
    if row.get("TIME") is not None:
        return str(row["TIME"])
    return f"reading {i + 1}"


def _status(P, row, state, i):
    P["clock"].markdown(
        f'<div class="statusbar"><span class="clock">\u25CF FARM TIME {_clock_label(row, i)}</span>'
        f'<span>state <b>{state}</b></span><span>1 reading / min</span></div>',
        unsafe_allow_html=True)


def stream_agent(df, obs, clf, pace, P):
    """Run the full agent loop over df, rendering into the placeholders in P.
    The thermal pulse data (k) comes from the data, never from the user."""
    conn = sqlite3.connect(eng.DB_PATH); conn.execute("DELETE FROM verified_logs"); conn.commit(); conn.close()
    obs._streak = 0
    cooldown = 0
    hist, last_event = [], None
    progress = st.progress(0.0)

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        hist.append({"t": i, "DO": row["DO"], "pH": row["pH"], "NH4_N": row["NH4_N"]})
        state, fired = "MONITOR", None

        if cooldown > 0:
            cooldown -= 1
            state = "COOLDOWN"
        else:
            window = df.iloc[max(0, i - WINDOW + 1): i + 1]
            is_anom = obs.score_row(window) if i > 0 else False
            if is_anom:
                state = "SUSPECT"
            if obs.update_debounce(is_anom):
                # ---- SUSPECT -> visible INTERROGATE beat (the ~30s pulse) ----
                state = "INTERROGATE"
                _status(P, row, state, i)
                P["chips"].markdown(chips_html("INTERROGATE"), unsafe_allow_html=True)
                P["banner"].markdown(banner_html(None, busy=True), unsafe_allow_html=True)
                P["cards"].markdown(cards_html(row), unsafe_allow_html=True)
                P["th_head"].markdown(thermal_busy_html(), unsafe_allow_html=True)
                time.sleep(1.3)
                # ---- the AI alone reads the k-value and decides ----
                fired = evaluate(row, float(row.get("k_value", 0.85)), clf)
                eng.log_decision(row, fired["k_est"], fired["final"], fired["conf"])
                last_event = fired
                obs._streak = 0
                cooldown = COOLDOWN_LIVE
                state = "VERDICT"

        _status(P, row, state, i)
        P["chips"].markdown(
            chips_html(state, last_event["severity"] if (state == "VERDICT" and last_event) else None),
            unsafe_allow_html=True)
        P["cards"].markdown(cards_html(row), unsafe_allow_html=True)
        P["chart"].altair_chart(param_chart(pd.DataFrame(hist).tail(240)), use_container_width=True)

        if fired is not None:
            P["banner"].markdown(banner_html(fired), unsafe_allow_html=True)
            P["th_head"].markdown(thermal_result_html(fired, _clock_label(row, i)), unsafe_allow_html=True)
            P["th_chart"].altair_chart(thermal_chart(fired["curve"], fired["severity"]), use_container_width=True)
            P["log"].dataframe(eng.fetch_logs(8), use_container_width=True, hide_index=True)
            time.sleep(2.0)
        elif state == "COOLDOWN":
            P["banner"].markdown(banner_html(last_event), unsafe_allow_html=True)
            time.sleep(pace)
        else:                                          # MONITOR / SUSPECT -> k blank, no curve
            P["banner"].markdown(banner_html(None), unsafe_allow_html=True)
            P["th_head"].markdown(thermal_idle_html(), unsafe_allow_html=True)
            P["th_chart"].empty()
            time.sleep(pace)

        progress.progress((i + 1) / len(df))

    P["log"].dataframe(eng.fetch_logs(12), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# TAB 1 - live monitor  (real-time, fixed one-per-minute cadence, no speed control)
# --------------------------------------------------------------------------- #
def tab_live(feed, obs, clf):
    start = st.button("\u25B6  Start live feed")
    P = build_console()

    first = feed.iloc[0].to_dict()
    _status(P, first, "MONITOR", 0)
    P["chips"].markdown(chips_html("MONITOR"), unsafe_allow_html=True)
    P["banner"].markdown(banner_html(None), unsafe_allow_html=True)
    P["cards"].markdown(cards_html(first), unsafe_allow_html=True)
    P["th_head"].markdown(thermal_idle_html(), unsafe_allow_html=True)

    if not start:
        st.info("Press **Start live feed** to watch the tank in real time, one reading per minute.")
        return

    stream_agent(feed, obs, clf, LIVE_PACE, P)
    st.success("Feed complete \u2014 every decision was logged offline first, then would sync to the cloud.")


# --------------------------------------------------------------------------- #
# TAB 2 - manual test console  (single reading  OR  data file)
# --------------------------------------------------------------------------- #
def tab_single(obs, clf):
    st.caption("Enter one water reading. The agent screens it against the SAMAQ limits and the tank's "
               "learned normal. A single typed reading has no probe attached, so it is taken at face "
               "value \u2014 to test whether a low reading is a real crisis or a fouled sensor, use "
               "'Run a data file', where the AI reads the probe's thermal pulse and decides for itself.")
    a, b, c, d = st.columns(4)
    do  = a.number_input("Dissolved O\u2082 (mg/L)", 0.0, 20.0, 6.10, 0.1)
    ph  = b.number_input("pH", 0.0, 14.0, 7.70, 0.1)
    nh4 = c.number_input("Ammonia NH\u2084 (mg/L)", 0.0, 50.0, 1.00, 0.1)
    temp = d.number_input("Temp (\u00b0C)", 0.0, 45.0, 26.8, 0.1)

    if st.button("Run reading"):
        row = {"TIME": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
               "DO": do, "pH": ph, "NH4_N": nh4, "Temp": temp}
        st.markdown(cards_html(row), unsafe_allow_html=True)
        level_off = any(abs(row[p] - obs.base_mean[p]) / obs.base_std[p] > obs.sigma
                        for p in eng.FEATURE_COLS)
        viol, text, _ = eng.diagnose_water(row)

        if not (level_off or viol):
            st.markdown(banner_html(None), unsafe_allow_html=True)
            st.info("Reading is within the tank's learned normal and the SAMAQ limits \u2014 nothing to flag.")
        elif viol:
            ev = {"severity": "crisis", "final": "WATER-QUALITY ALERT",
                  "action": "A parameter is outside the SAMAQ limits.", "detail": text}
            st.markdown(banner_html(ev), unsafe_allow_html=True)
            st.caption("Whether this is genuinely bad water or a fouled probe can only be confirmed by a "
                       "thermal pulse \u2014 run a data file with the probe log to let the AI decide.")
        else:
            ev = {"severity": "suppress", "final": "ANOMALY \u2014 off the learned normal",
                  "action": "Within SAMAQ limits but drifting from the tank's normal range.",
                  "detail": "A thermal interrogation is needed to confirm whether the probe or the water is the cause."}
            st.markdown(banner_html(ev), unsafe_allow_html=True)


def tab_file(obs, clf):
    speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
    pace = {"Slow": 0.40, "Normal": 0.15, "Fast": 0.04}[speed]
    st.caption("Upload a readings file (CSV). Include a k_value column \u2014 the probe's logged thermal "
               "pulse \u2014 so the agent can tell a sensor fault from a real crisis on its own. The AI "
               "evaluates every k-value itself; you never label the rows. (Thermal pulses are simulated in this prototype.)")
    up = st.file_uploader("Readings CSV", type=["csv"])
    run = st.button("Run file")

    if up is not None and run:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            return
        missing = [c for c in eng.FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"File is missing required columns: {missing}. It needs DO, pH, NH4_N, Temp "
                     f"(and ideally a k_value column).")
            return
        if "k_value" not in df.columns:
            st.warning("No k_value column found \u2014 the agent will assume a clean probe, so it cannot "
                       "detect a sensor fault. Add a k_value column to enable fault-vs-crisis discrimination.")
        P = build_console()
        stream_agent(df, obs, clf, pace, P)
        st.success("File processed \u2014 every decision logged offline.")
    else:
        st.info("Upload a CSV (for example your sensor_baseline_fail.csv or pool_crisis.csv) and press Run file.")


def tab_manual(obs, clf):
    mode = st.radio("Test input", ["Single reading", "Run a data file"], horizontal=True)
    st.write("")
    if mode == "Single reading":
        tab_single(obs, clf)
    else:
        tab_file(obs, clf)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="RAS Sentinel", page_icon="\U0001F41F", layout="wide")
    inject_css()
    feed, obs, clf = get_models()

    st.markdown(
        '<div class="hdr"><div>'
        '<div class="title">RAS&nbsp;<small>SENTINEL</small></div>'
        f'<div class="sub">{FARM} \u00b7 {TANK}</div></div>'
        '</div>', unsafe_allow_html=True)

    live, manual = st.tabs(["  Live monitor  ", "  Manual test console  "])
    with live:
        tab_live(feed, obs, clf)
    with manual:
        tab_manual(obs, clf)


if __name__ == "__main__":
    main()
