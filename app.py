"""
FactoryPulse AI - Streamlit dashboard for production bottleneck prediction.

Main file for Streamlit Cloud: app.py  (keep factorypulse_bottleneck.py next to it)
Tabs: Live monitor | All stations | Prediction quality | Model insights |
      Scenario comparison | Data
"""
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import factorypulse_bottleneck as fp

st.set_page_config(page_title="FactoryPulse AI", page_icon="🏭", layout="wide")

BLUE, RED, AMBER, GREEN, GREY, DARK = "#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#6b7280", "#111827"
STATUS_COLORS = {"NORMAL": GREEN, "WARNING": AMBER, "CRITICAL": RED, "STOPPED": DARK, "IDLE": GREY}
SCENARIOS = {k: v for k, v in fp.FactorySimulator.SCENARIOS.items() if k != "scenario_f"}
CFG = fp.config
RISK_THRESHOLD = 0.5
FEATURE_HELP = {
    "rpm": "Current motor speed of station 2",
    "cycle_time": "Seconds needed per unit (rises as the machine slows)",
    "throughput": "Units per minute leaving station 2",
    "utilization": "How much of the nominal capacity is in use",
    "queue_length": "Units waiting in front of station 2",
    "vibration": "Vibration level (mm/s)",
    "temperature": "Motor temperature (°C)",
    "current": "Motor current (A)",
    "demand_rate": "Incoming demand (units/min)",
    "line_throughput": "Units per minute leaving the whole line",
    "total_wip": "Total units waiting in all queues",
    "rpm_rolling_mean": "Average RPM over the last 10 s",
    "rpm_rolling_std": "RPM variability over the last 10 s",
    "rpm_trend": "How fast RPM is rising or falling",
    "tp_trend": "How fast throughput is rising or falling",
    "queue_growth_rate": "How fast the queue is growing",
    "tp_to_demand_ratio": "Throughput divided by demand (below 1 = falling behind)",
}


# =============================================================================
# Data: model + simulation
# =============================================================================
@st.cache_resource(show_spinner="Training model on first run (about 10 seconds)...")
def ensure_model() -> bool:
    """Train the model once if no saved model exists (Streamlit Cloud starts empty)."""
    if not CFG.ML_MODEL_PATH.exists():
        random.seed(42)
        np.random.seed(42)
        fp.cmd_train(42)
    return True


@st.cache_data(show_spinner=False)
def load_model_info():
    """Feature importances (from the model) and training metrics (from metadata)."""
    import json

    import joblib

    pkg = joblib.load(CFG.ML_MODEL_PATH)
    importances = dict(zip(pkg["feature_columns"], map(float, pkg["model"].feature_importances_)))
    meta = None
    if CFG.ML_METADATA_PATH.exists():
        meta = json.loads(CFG.ML_METADATA_PATH.read_text(encoding="utf-8"))
    return importances, meta


def add_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Actual outcome: did a bottleneck really occur in [t+10 s, t+60 s]? (same rule as training)"""
    h = CFG.ML_PREDICTION_HORIZON_SEC
    q, r, n = df["Queue"].to_numpy(), df["RPM"].to_numpy(), len(df)
    labels = np.full(n, np.nan)
    for i in range(n):
        s, e = min(n, i + 10), min(n, i + h + 1)
        if s < n and e - s >= 10:
            labels[i] = float(q[s:e].max() >= CFG.ALERT_QUEUE_WARNING_THRESHOLD or r[s:e].min() < 900.0)
    df["Actual bottleneck"] = labels
    return df


@st.cache_data(show_spinner=False)
def run_simulation(scenario: str, steps: int, seed: int, demand=None) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    predictor = fp.RealTimePredictor()
    sim = fp.FactorySimulator()
    sim.set_scenario(scenario)
    if demand is not None:
        sim.demand_rate = float(demand)
    rows = []
    for t in range(steps):
        out = sim.step(dt_override=1.0)
        line = out["processed"]
        predictor.update_line_metrics(line.model_dump())
        s1, s2, s3 = out["station01"], out["station02"], out["station03"]
        pred = predictor.process_reading(s2.model_dump())
        rows.append({
            "time_s": t,
            "RPM": s2.rpm, "Queue": s2.queue_length, "Throughput": s2.throughput,
            "Utilization": s2.utilization, "Temperature": s2.temperature,
            "Vibration": s2.vibration, "Current": s2.current, "Status": s2.status,
            "S1 RPM": s1.rpm, "S3 RPM": s3.rpm,
            "S1 Queue": s1.queue_length, "S2 Queue": s2.queue_length, "S3 Queue": s3.queue_length,
            "Demand": line.demand_rate, "Line throughput": line.line_throughput,
            "Total WIP": line.total_wip, "Line efficiency": line.line_efficiency,
            "Line bottleneck": line.bottleneck_station or "none",
            "Bottleneck risk": pred.risk_probability,
            "Bottleneck predicted": pred.bottleneck_predicted,
        })
    return add_ground_truth(pd.DataFrame(rows))


# =============================================================================
# Helpers
# =============================================================================
def alert_segments(df: pd.DataFrame) -> pd.DataFrame:
    t, f, r = df["time_s"].to_numpy(), df["Bottleneck predicted"].to_numpy(), df["Bottleneck risk"].to_numpy()
    segs, start = [], None
    for i in range(len(df)):
        if f[i] and start is None:
            start = i
        if start is not None and (not f[i] or i == len(df) - 1):
            end = i if f[i] else i - 1
            segs.append({"Start (s)": int(t[start]), "End (s)": int(t[end]),
                         "Duration (s)": int(t[end] - t[start] + 1),
                         "Peak risk": round(float(r[start:end + 1].max()), 3)})
            start = None
    return pd.DataFrame(segs)


def first_event_time(df: pd.DataFrame):
    m = (df["Queue"] >= CFG.ALERT_QUEUE_WARNING_THRESHOLD) | (df["RPM"] < 900.0)
    return int(df.loc[m, "time_s"].iloc[0]) if m.any() else None


def first_alert_time(df: pd.DataFrame):
    m = df["Bottleneck predicted"]
    return int(df.loc[m, "time_s"].iloc[0]) if m.any() else None


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# Chart builders (each returns a matplotlib Figure)
# =============================================================================
def _new(h=3.2, w=9.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Time (s)")
    return fig, ax


def _shade(ax, df, on=True):
    if not on:
        return
    f, t = df["Bottleneck predicted"].to_numpy(), df["time_s"].to_numpy()
    start = None
    for i in range(len(f)):
        if f[i] and start is None:
            start = t[i]
        if start is not None and (not f[i] or i == len(f) - 1):
            ax.axvspan(start, t[i], color=RED, alpha=0.08, lw=0)
            start = None


def fig_risk(df, shade=True):
    fig, ax = _new(3.6)
    ax.plot(df["time_s"], df["Bottleneck risk"], color=BLUE, lw=2, label="Predicted risk")
    ax.fill_between(df["time_s"], df["Bottleneck risk"], color=BLUE, alpha=0.12)
    ax.axhline(RISK_THRESHOLD, color=RED, ls="--", lw=1, label="Alert threshold (0.5)")
    ax.step(df["time_s"], df["Actual bottleneck"], where="post", color=GREEN, lw=1.5, alpha=0.8,
            label="Actual bottleneck ahead (ground truth)")
    _shade(ax, df, shade)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Risk (0-1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    return fig


def fig_rpm(df, shade=True):
    fig, ax = _new()
    base = CFG.STATION2_BASE_RPM
    ax.plot(df["time_s"], df["RPM"], color=BLUE, lw=2, label="Station 2 RPM")
    ax.axhline(base * CFG.ALERT_RPM_WARNING_RATIO, color=AMBER, ls="--", lw=1, label="Warning (70%)")
    ax.axhline(base * CFG.ALERT_RPM_CRITICAL_RATIO, color=RED, ls="--", lw=1, label="Critical (40%)")
    ax.axhline(900, color=GREY, ls=":", lw=1, label="Bottleneck label limit (900)")
    _shade(ax, df, shade)
    ax.set_ylabel("RPM")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return fig


def fig_queue(df, shade=True):
    fig, ax = _new()
    ax.plot(df["time_s"], df["Queue"], color=BLUE, lw=2, label="Station 2 queue")
    ax.axhline(CFG.ALERT_QUEUE_WARNING_THRESHOLD, color=AMBER, ls="--", lw=1, label="Warning (15)")
    ax.axhline(CFG.ALERT_QUEUE_CRITICAL_THRESHOLD, color=RED, ls="--", lw=1, label="Critical (30)")
    _shade(ax, df, shade)
    ax.set_ylabel("Units waiting")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return fig


def fig_throughput(df):
    fig, ax = _new()
    ax.plot(df["time_s"], df["Throughput"], color=BLUE, lw=2, label="Station 2 throughput")
    ax.plot(df["time_s"], df["Line throughput"], color=GREEN, lw=2, label="Line throughput (exit)")
    ax.plot(df["time_s"], df["Demand"], color=RED, ls="--", lw=1.5, label="Demand")
    ax.set_ylabel("Units / min")
    ax.legend(fontsize=8, frameon=False)
    return fig


def fig_sensors(df):
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=True)
    for ax, col, color, lab in zip(axes, ["Temperature", "Vibration", "Current"],
                                   [RED, AMBER, BLUE], ["Temp (°C)", "Vibration (mm/s)", "Current (A)"]):
        ax.plot(df["time_s"], df[col], color=color, lw=1.8)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def fig_status(df):
    fig, ax = plt.subplots(figsize=(9, 1.0))
    ax.bar(df["time_s"], 1, width=1.0, color=[STATUS_COLORS.get(s, GREY) for s in df["Status"]])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(df["time_s"].min(), df["time_s"].max())
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[Patch(color=c, label=s.title()) for s, c in STATUS_COLORS.items()
                       if s in set(df["Status"])], ncol=5, fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 2.0))
    return fig


def fig_multi(df, cols, labels, ylabel, colors=(BLUE, AMBER, GREEN), hlines=()):
    fig, ax = _new()
    for c, lab, col in zip(cols, labels, colors):
        ax.plot(df["time_s"], df[c], lw=2, color=col, label=lab)
    for y, col, lab in hlines:
        ax.axhline(y, color=col, ls="--", lw=1, label=lab)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    return fig


def fig_wip(df):
    fig, ax = _new()
    ax.stackplot(df["time_s"], df["S1 Queue"], df["S2 Queue"], df["S3 Queue"],
                 labels=["Station 1 (Cutting)", "Station 2 (Drilling)", "Station 3 (Packaging)"],
                 colors=[BLUE, AMBER, GREEN], alpha=0.8)
    ax.set_ylabel("Work in progress (units)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return fig


def fig_efficiency(df):
    fig, ax = _new()
    ax.plot(df["time_s"], df["Line efficiency"] * 100, color=GREEN, lw=2)
    ax.fill_between(df["time_s"], df["Line efficiency"] * 100, color=GREEN, alpha=0.15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Line efficiency (%)")
    return fig


def fig_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    names = ["Normal", "Bottleneck"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    top = np.max(cm) if np.max(cm) > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > top / 2 else "black")
    fig.tight_layout()
    return fig


def fig_importance(importances, top=12):
    items = sorted(importances.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_model_compare(meta):
    base, rf = meta["baseline_comparison"], meta
    b = [base["baseline_accuracy"], base["baseline_f1"]]
    r = [rf["accuracy"], rf["f1_score"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x - 0.18, b, 0.36, label="Rule-based baseline", color=GREY)
    ax.bar(x + 0.18, r, 0.36, label="Random Forest", color=BLUE)
    for xi, v in zip(x - 0.18, b):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.18, r):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Accuracy", "F1 score"])
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_hist(df):
    d = df.dropna(subset=["Actual bottleneck"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(d.loc[d["Actual bottleneck"] == 0, "Bottleneck risk"], bins=bins, alpha=0.7, color=GREEN,
            label="No bottleneck ahead")
    ax.hist(d.loc[d["Actual bottleneck"] == 1, "Bottleneck risk"], bins=bins, alpha=0.7, color=RED,
            label="Bottleneck ahead")
    ax.axvline(RISK_THRESHOLD, color=DARK, ls="--", lw=1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_scatter(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sc = ax.scatter(df["RPM"], df["Bottleneck risk"], c=df["Queue"], cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, label="Queue length")
    ax.set_xlabel("Station 2 RPM")
    ax.set_ylabel("Predicted risk")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_scenarios(results, col, ylabel, hline=None):
    fig, ax = _new(3.6)
    for (key, d), c in zip(results.items(), plt.cm.tab10.colors):
        ax.plot(d["time_s"], d[col], lw=2, color=c, label=SCENARIOS[key].split(":")[0])
    if hline is not None:
        ax.axhline(hline, color=DARK, ls="--", lw=1)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=8, frameon=False)
    return fig


# =============================================================================
# Page
# =============================================================================
st.title("🏭 FactoryPulse AI - Bottleneck Prediction")
st.caption("Simulated 3-station factory (Cutting → Drilling → Packaging). The model predicts whether "
           "a bottleneck will occur within the next 60 seconds, using only past sensor readings.")

ensure_model()

with st.sidebar:
    st.header("Simulation settings")
    scenario = st.selectbox("Scenario", list(SCENARIOS), index=1, format_func=lambda k: SCENARIOS[k])
    steps = st.slider("Duration (seconds)", 100, 300, 180, step=10)
    seed = int(st.number_input("Random seed", 0, 9999, 42))
    override = st.checkbox("Override demand rate")
    demand = st.slider("Demand (units/min)", 8.0, 30.0, 15.0, step=0.5) if override else None
    shade = st.checkbox("Shade predicted-alert periods", value=True)
    st.caption("Change any setting to re-run. Results are cached for repeat views.")

df = run_simulation(scenario, steps, seed, demand)
st.subheader(SCENARIOS[scenario])

# ---- KPI row ----------------------------------------------------------------
alert_t, event_t = first_alert_time(df), first_event_time(df)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("First warning", f"{alert_t} s" if alert_t is not None else "none")
k2.metric("Actual bottleneck at", f"{event_t} s" if event_t is not None else "none")
k3.metric("Peak risk", f"{df['Bottleneck risk'].max():.0%}")
k4.metric("Min RPM", f"{df['RPM'].min():.0f}")
k5.metric("Max queue", f"{int(df['Queue'].max())}")
k6.metric("Time flagged", f"{df['Bottleneck predicted'].mean():.0%}")
if alert_t is not None and event_t is not None:
    lead = event_t - alert_t
    if lead > 0:
        st.success(f"Early warning: the model raised an alert {lead} s before the first real bottleneck.")
    else:
        st.warning("The first alert came at or after the first real bottleneck.")
elif alert_t is not None:
    st.info("The model raised an alert but no real bottleneck occurred in this run (possible false alarm).")
elif event_t is not None:
    st.error("A real bottleneck occurred without any warning (missed detection).")
else:
    st.success("Stable run: no bottleneck occurred and none was predicted.")

tabs = st.tabs(["📈 Live monitor", "🏭 All stations", "🎯 Prediction quality",
                "🧠 Model insights", "⚖️ Scenario comparison", "📄 Data"])

# ---- Tab 1: live monitor ------------------------------------------------------
with tabs[0]:
    st.markdown("**Bottleneck risk vs. what actually happened** (red shading = model says bottleneck ahead)")
    show(fig_risk(df, shade))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Station 2 motor speed**")
        show(fig_rpm(df, shade))
        st.markdown("**Throughput vs. demand**")
        show(fig_throughput(df))
    with c2:
        st.markdown("**Station 2 queue**")
        show(fig_queue(df, shade))
        st.markdown("**Machine status timeline**")
        show(fig_status(df))
    st.markdown("**Motor condition sensors**")
    show(fig_sensors(df))
    st.markdown("**Alert log**")
    segs = alert_segments(df)
    if len(segs):
        st.dataframe(segs)
    else:
        st.caption("No alerts were raised in this run.")

# ---- Tab 2: all stations ------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Queue per station**")
        show(fig_multi(df, ["S1 Queue", "S2 Queue", "S3 Queue"], ["Station 1", "Station 2", "Station 3"],
                       "Units waiting",
                       hlines=[(CFG.ALERT_QUEUE_WARNING_THRESHOLD, RED, "Warning (15)")]))
        st.markdown("**Work in progress (stacked)**")
        show(fig_wip(df))
    with c2:
        st.markdown("**Motor speed per station**")
        show(fig_multi(df, ["S1 RPM", "RPM", "S3 RPM"], ["Station 1", "Station 2", "Station 3"], "RPM"))
        st.markdown("**Line efficiency**")
        show(fig_efficiency(df))
    st.markdown("**Where the line says the bottleneck is** (seconds per station)")
    st.bar_chart(df["Line bottleneck"].value_counts())

# ---- Tab 3: prediction quality -------------------------------------------------
with tabs[2]:
    lab = df.dropna(subset=["Actual bottleneck"])
    if len(lab) < 10:
        st.info("Run a longer simulation (at least 80 s) to score the predictions.")
    else:
        y, p = lab["Actual bottleneck"].astype(int), lab["Bottleneck predicted"].astype(int)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy (this run)", f"{accuracy_score(y, p):.1%}")
        m2.metric("Precision", f"{precision_score(y, p, zero_division=0):.1%}")
        m3.metric("Recall", f"{recall_score(y, p, zero_division=0):.1%}")
        m4.metric("F1 score", f"{f1_score(y, p, zero_division=0):.3f}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix (this run)**")
            show(fig_confusion(confusion_matrix(y, p, labels=[0, 1]), f"{len(lab)} scored seconds"))
        with c2:
            st.markdown("**Risk score distribution**")
            show(fig_hist(df))
        st.markdown("**Risk vs. RPM** (colour = queue length)")
        show(fig_scatter(df))
        st.caption("The last ~60 s of every run cannot be scored because the future is unknown.")

# ---- Tab 4: model insights -----------------------------------------------------
with tabs[3]:
    importances, meta = load_model_info()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Which signals matter most**")
        show(fig_importance(importances))
    with c2:
        if meta:
            st.markdown("**Random Forest vs. rule-based baseline** (held-out runs)")
            show(fig_model_compare(meta))
    if meta:
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{meta['accuracy']:.1%}")
        b.metric("Precision", f"{meta['precision']:.1%}")
        c.metric("Recall", f"{meta['recall']:.1%}")
        d.metric("F1", f"{meta['f1_score']:.3f}")
        e.metric("ROC-AUC", f"{meta['roc_auc']:.3f}" if meta.get("roc_auc") is not None else "n/a")
        st.markdown("**Confusion matrix on held-out test runs**")
        show(fig_confusion(np.array(meta["confusion_matrix"]),
                           f"{meta['test_samples']} test samples ({meta['model_version']})"))
    with st.expander("What do the features mean?"):
        st.table(pd.DataFrame({"Feature": list(FEATURE_HELP), "Meaning": list(FEATURE_HELP.values())}))

# ---- Tab 5: scenario comparison ------------------------------------------------
with tabs[4]:
    results = {k: run_simulation(k, steps, seed, demand) for k in SCENARIOS}
    summary = pd.DataFrame([{
        "Scenario": SCENARIOS[k].split(": ")[1],
        "First warning (s)": first_alert_time(d),
        "Real bottleneck (s)": first_event_time(d),
        "Peak risk": round(float(d["Bottleneck risk"].max()), 3),
        "Min RPM": round(float(d["RPM"].min()), 1),
        "Max queue": int(d["Queue"].max()),
        "Time flagged (%)": round(100 * float(d["Bottleneck predicted"].mean()), 1),
    } for k, d in results.items()])
    st.dataframe(summary)
    st.markdown("**Predicted risk, all scenarios**")
    show(fig_scenarios(results, "Bottleneck risk", "Risk (0-1)", RISK_THRESHOLD))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Station 2 RPM**")
        show(fig_scenarios(results, "RPM", "RPM"))
    with c2:
        st.markdown("**Station 2 queue**")
        show(fig_scenarios(results, "Queue", "Units waiting", CFG.ALERT_QUEUE_WARNING_THRESHOLD))

# ---- Tab 6: data ---------------------------------------------------------------
with tabs[5]:
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"factorypulse_{scenario}.csv", mime="text/csv")
