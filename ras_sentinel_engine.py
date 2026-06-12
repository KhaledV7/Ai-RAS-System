"""
RAS SENTINEL - ras_sentinel_engine.py  (STEP 1: The Core AI Engine)
============================================================
A hardware-agnostic Edge AI engine for Recirculating Aquaculture Systems (RAS).

Core idea: trust no single reading. An Observer (unsupervised) watches the water
and only SUSPECTS a problem; a Diagnostician then actively interrogates the sensor
(thermal pulse) to decide whether a low reading is a REAL water crisis or just a
FOULED probe -- before raising any alarm.

Design choices (deliberate, for the judges):
  * NO neural networks -> lightweight + fully explainable on an edge device.
  * Observer uses ENGINEERED rate-of-change features, because real RAS telemetry is
    nearly flat and slow drift is invisible point-by-point.
  * The thermal layer runs in SIMULATION / DIGITAL-TWIN mode for now. We have no real
    thermal measurements yet; get_thermal_curve() is the single swap point for real data.

Author: RAS Sentinel team
"""

import sqlite3
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. SAMAQ / BAP CONSTANTS  (extracted from the BAP Effluent Criteria, RAS column)
# ---------------------------------------------------------------------------
# These are the certified discharge limits we build the Diagnostician against.
SAMAQ = {
    "DO_MIN":   5.0,    # Dissolved oxygen: "more than 5 mg/L"
    "PH_MIN":   6.0,    # pH range 6.0 - 9.5
    "PH_MAX":   9.5,
    "NH4_MAX":  5.0,    # Total ammonia nitrogen: "less than 5 mg/L"
}

# NOTE: BAP gives NO temperature limit. The band below is NOT a BAP rule.
# Species: NILE TILAPIA (Oreochromis niloticus) -- the most common RAS finfish in
# Saudi Arabia. Preferred warm-water range, which matches our ~26.8 C data.
# Editable: change these two numbers if the farm uses a different species.
TEMP_COMFORT = {"TEMP_MIN": 22.0, "TEMP_MAX": 32.0}   # Nile tilapia comfort band

DB_PATH = "edge_telemetry.db"
BASELINE_ROWS = 2000          # rows of real context used to train the Observer
DEBOUNCE_N = 3                # an anomaly must persist this many readings to escalate


# ---------------------------------------------------------------------------
# 2. DATABASE  (offline edge storage)
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH):
    """Create the local SQLite store for verified decisions (offline edge log)."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verified_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            do         REAL,
            ph         REAL,
            temp       REAL,
            nh4        REAL,
            k_value    REAL,
            ai_verdict TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def log_decision(row: dict, k_value, verdict: str, confidence: float,
                 db_path: str = DB_PATH):
    """Persist one verified AI decision before any cloud sync."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO verified_logs
           (timestamp, do, ph, temp, nh4, k_value, ai_verdict, confidence)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            row.get("TIME", datetime.datetime.now().isoformat(timespec="seconds")),
            float(row["DO"]), float(row["pH"]), float(row["Temp"]), float(row["NH4_N"]),
            None if k_value is None else float(k_value),
            verdict, float(confidence),
        ),
    )
    conn.commit()
    conn.close()


def fetch_logs(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """Read the most recent verified decisions back out (used by the dashboard)."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM verified_logs ORDER BY id DESC LIMIT {int(limit)}", conn
        )
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# 3. DATA SANITATION
# ---------------------------------------------------------------------------
def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sensor warm-up garbage so it never poisons the baseline.
    Real symptom in our CSV: first rows show pH == 0 and NH4 spikes of 8 / 23."""
    df = df.copy()
    # startup warm-up garbage = physically implausible sensor states. A running RAS
    # never reads pH < 5, and our real crises push pH UP and ammonia to ~9, so these
    # floors/ceilings only catch boot noise, never real faults.
    mask_garbage = (df["pH"] < 5) | (df["NH4_N"] > 15) | (df["DO"] <= 0) | (df["DO"] > 20)
    removed = int(mask_garbage.sum())
    df = df[~mask_garbage].reset_index(drop=True)
    if removed:
        print(f"[sanitize] removed {removed} warm-up/garbage row(s)")
    return df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING  (rate-of-change, because raw data is nearly flat)
# ---------------------------------------------------------------------------
FEATURE_COLS = ["DO", "pH", "NH4_N", "Temp"]


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Turn raw readings into level + short-window slope features.
    Slow drift is invisible point-by-point but obvious in the slope."""
    feats = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        feats[f"{c}_level"] = df[c]
        feats[f"{c}_slope"] = df[c].diff().rolling(window, min_periods=1).mean()
    return feats.fillna(0.0)


# ---------------------------------------------------------------------------
# 5. THE OBSERVER  (unsupervised anomaly detector + debounce)
# ---------------------------------------------------------------------------
class Observer:
    """Learns the tank's normal rhythm and flags persistent deviations only.

    Two unsupervised detectors, BOTH learned from the baseline:
      1. Isolation Forest -> multivariate PATTERN anomalies (e.g. the toxic spike).
      2. Statistical control limits (n-sigma per parameter) -> single-parameter LEVEL
         drift that a multivariate model averages away (e.g. the slow DO sag).
    A reading is suspicious if EITHER fires. Defense in depth, no neural nets."""

    def __init__(self, contamination: float = 0.01, debounce_n: int = DEBOUNCE_N,
                 sigma: float = 5.0):
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        self.scaler = StandardScaler()
        self.debounce_n = debounce_n
        self.sigma = sigma
        self.base_mean = None
        self.base_std = None
        self._streak = 0
        self.trained = False

    def train(self, baseline_df: pd.DataFrame):
        feats = build_features(baseline_df)
        X = self.scaler.fit_transform(feats.values)
        self.model.fit(X)
        # per-parameter control limits from the raw baseline values
        self.base_mean = baseline_df[FEATURE_COLS].mean()
        self.base_std = baseline_df[FEATURE_COLS].std().replace(0, 1e-6)
        self.trained = True

    def score_row(self, recent_df: pd.DataFrame) -> bool:
        """Return True if the latest row is anomalous. recent_df = a small trailing
        window ending at the row under test (needed to compute the slope)."""
        if not self.trained:
            raise RuntimeError("Observer not trained")
        feats = build_features(recent_df)
        X = self.scaler.transform(feats.values)
        pattern_anom = self.model.predict(X)[-1] == -1            # Isolation Forest
        row = recent_df.iloc[-1]
        z = (row[FEATURE_COLS] - self.base_mean).abs() / self.base_std
        level_anom = bool((z > self.sigma).any())                 # control limits
        return bool(pattern_anom or level_anom)

    def update_debounce(self, is_anomaly: bool) -> bool:
        """Escalate to the Diagnostician only after N consecutive anomalies."""
        self._streak = self._streak + 1 if is_anomaly else 0
        return self._streak >= self.debounce_n


# ---------------------------------------------------------------------------
# 6. THE DIAGNOSTICIAN  (rules engine -> self-explaining verdict)
# ---------------------------------------------------------------------------
def diagnose_water(row: dict):
    """Check a reading against the SAMAQ/BAP limits.
    Returns (is_violation, human_text, confidence)."""
    problems = []
    if row["DO"] < SAMAQ["DO_MIN"]:
        problems.append(f"DO {row['DO']:.2f} mg/L is BELOW the SAMAQ minimum of "
                        f"{SAMAQ['DO_MIN']} mg/L")
    if row["pH"] < SAMAQ["PH_MIN"] or row["pH"] > SAMAQ["PH_MAX"]:
        problems.append(f"pH {row['pH']:.2f} is OUTSIDE the SAMAQ range "
                        f"{SAMAQ['PH_MIN']}-{SAMAQ['PH_MAX']}")
    if row["NH4_N"] > SAMAQ["NH4_MAX"]:
        problems.append(f"Total ammonia {row['NH4_N']:.2f} mg/L EXCEEDS the SAMAQ "
                        f"max of {SAMAQ['NH4_MAX']} mg/L")
    # Temperature: NOT a SAMAQ rule -- flagged against the editable comfort band.
    if row["Temp"] < TEMP_COMFORT["TEMP_MIN"] or row["Temp"] > TEMP_COMFORT["TEMP_MAX"]:
        problems.append(f"Temp {row['Temp']:.2f} C is outside the species comfort band "
                        f"{TEMP_COMFORT['TEMP_MIN']}-{TEMP_COMFORT['TEMP_MAX']} C "
                        f"(note: not a BAP limit)")

    if not problems:
        return False, "All parameters within SAMAQ limits.", 0.0
    # crude confidence: more simultaneous violations -> higher confidence
    confidence = min(1.0, 0.5 + 0.25 * (len(problems) - 1))
    return True, " | ".join(problems), confidence


# ---------------------------------------------------------------------------
# 7. PIML THERMAL LAYER  (Newton's Law of Cooling -> SVM)  *** SIMULATION MODE ***
# ---------------------------------------------------------------------------
PULSE_SECONDS = 30
PULSE_HZ = 10
N_SAMPLES = PULSE_SECONDS * PULSE_HZ        # 300 points per pulse
DELTA_T = 5.0                               # heat pulse raises probe ~5 C


def _newton_curve(k: float, noise: float = 0.05, n: int = N_SAMPLES) -> np.ndarray:
    """Newton's Law of Cooling: T(t) = dT * exp(-k t). High k = fast cooling (clean,
    water carries heat away). Low k = slow cooling (fouled, heat trapped)."""
    t = np.linspace(0, 3, n)
    curve = DELTA_T * np.exp(-k * t)
    curve += np.random.normal(0, noise, n)
    return curve


def bootstrap_thermal_classifier(seed: int = 7):
    """Self-train an SVM on boot from synthetic Newton's-Law curves.
    *** SIMULATION MODE -- field calibration with real sonde data pending. ***"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(100):                              # CLEAN probes -> fast cooling
        k = rng.uniform(0.7, 1.0)
        X.append(_newton_curve(k)); y.append("CLEAN")
    for _ in range(100):                              # FOULED probes -> slow cooling
        k = rng.uniform(0.05, 0.25)
        X.append(_newton_curve(k)); y.append("FOULED")
    clf = SVC(kernel="rbf", probability=True, random_state=seed)
    clf.fit(np.array(X), np.array(y))
    return clf


def get_thermal_curve(k_value: float) -> np.ndarray:
    """SINGLE SWAP POINT. Today: simulate a curve from k. Tomorrow: return the real
    10 Hz / 30 s thermistor reading from the sonde. Nothing else changes."""
    return _newton_curve(k_value)        # <-- replace body with real sensor read


def classify_pulse(clf, curve: np.ndarray):
    """Classify a 300-point thermal pulse. Returns (verdict, confidence, est_k)."""
    label = clf.predict([curve])[0]
    proba = float(np.max(clf.predict_proba([curve])[0]))
    # estimate k from the curve for logging (least-squares on log decay)
    t = np.linspace(0, 3, len(curve))
    pos = np.clip(curve, 1e-3, None)
    k_est = float(-np.polyfit(t, np.log(pos), 1)[0])
    verdict = ("HARDWARE FAULT (biofouling) - suppress water alarm"
               if label == "FOULED"
               else "WATER CRISIS confirmed - sensor verified clean")
    return verdict, proba, k_est


# ---------------------------------------------------------------------------
# 8. SELF-TEST  (run this file directly to verify Step 1 before Step 2)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("RAS SENTINEL - Step 1 self-test")
    print("=" * 64)

    init_db()
    print(f"[db] initialised {DB_PATH}")

    df = pd.read_csv("paramters.Csv")
    df = sanitize(df)
    print(f"[data] {len(df)} clean rows loaded")

    obs = Observer()
    obs.train(df.iloc[:BASELINE_ROWS])
    print(f"[observer] trained on {BASELINE_ROWS} baseline rows")

    clf = bootstrap_thermal_classifier()
    print("[thermal] SVM bootstrapped on synthetic curves (SIMULATION MODE)")

    print("\n--- Diagnostician demo ---")
    samples = {
        "healthy row":        {"TIME": "demo", "DO": 6.1, "pH": 7.7, "NH4_N": 1.0, "Temp": 26.8},
        "low-DO crisis":      {"TIME": "demo", "DO": 3.0, "pH": 7.7, "NH4_N": 1.0, "Temp": 26.8},
        "ammonia spike":      {"TIME": "demo", "DO": 6.0, "pH": 8.9, "NH4_N": 9.5, "Temp": 26.8},
    }
    for name, row in samples.items():
        viol, text, conf = diagnose_water(row)
        print(f"  [{name}] violation={viol} conf={conf:.2f} -> {text}")

    print("\n--- Thermal interrogation demo ---")
    for label, k in [("clean probe (k=0.85)", 0.85), ("fouled probe (k=0.12)", 0.12)]:
        curve = get_thermal_curve(k)
        verdict, conf, k_est = classify_pulse(clf, curve)
        print(f"  [{label}] -> {verdict}  (conf={conf:.2f}, k_est={k_est:.2f})")
        log_decision(samples["low-DO crisis"], k_est, verdict, conf)

    print("\n--- Database proof (latest logs) ---")
    print(fetch_logs(5).to_string(index=False))
    print("\nStep 1 OK.")
