"""
app.py — the fiberqc web app.

    streamlit run app.py

A thin layer over the fiberqc package: it loads a recording, runs the multiverse,
and shows where the researcher's own pipeline sits in it. All the science lives in
fiberqc; nothing is recomputed here.
"""

import os
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")          # must precede any pyplot import: Streamlit is
                               # multi-threaded and the GUI backend segfaults there

import numpy as np
import streamlit as st

import fiberqc as fqc

st.set_page_config(page_title="fiberqc", layout="wide")

# ----------------------------------------------------------------- style
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
                 sans-serif;
  }
  .tagline {color:#5b6470; font-size:1.02rem; margin:-.2rem 0 1.4rem 0;
            max-width: 62ch; line-height:1.5;}
  .verdict {padding: 1.05rem 1.25rem; border-radius: 10px; margin: .2rem 0 1.6rem 0;}
  .flip   {background:#fdeef0; border-left: 5px solid #d1495b;}
  .robust {background:#eaf2f8; border-left: 5px solid #1b6ca8;}
  .null   {background:#f2f3f5; border-left: 5px solid #8a929b;}
  .verdict h3 {margin: 0 0 .35rem 0; font-size: 1.12rem; color:#2b2f33;
               font-weight:700;}
  .verdict p  {margin: 0; color:#40474e; font-size: .95rem; line-height:1.5;}
  .stat {font-size: 1.85rem; font-weight: 700; line-height: 1.1; color:#2b2f33;}
  .stat.alarm {color:#d1495b;}
  .statlab {font-size: .72rem; color:#8a929b; text-transform: uppercase;
            letter-spacing: .06em; font-weight:600; margin-bottom:.15rem;}
</style>
""", unsafe_allow_html=True)

head_l, head_r = st.columns([3, 1])
with head_l:
    if os.path.exists("logo.svg"):
        st.image("logo.svg", width=280)
    else:
        st.title("fiberqc")
with head_r:
    st.markdown(
        '<div style="text-align:right;padding-top:1.6rem;font-size:.85rem">'
        '<a href="https://demiregeortac666.github.io/fiberqc/" '
        'style="color:#1b6ca8;text-decoration:none">API documentation ↗</a></div>',
        unsafe_allow_html=True)

st.markdown('<div class="tagline">Quality control for fiber photometry — is your '
            "result real, or an artifact of the preprocessing choices you didn't "
            'know you were making?</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Your recording")
    demo = st.selectbox(
        "Use a bundled example, or upload your own",
        ["— upload my own —",
         "Akam NAc dLight (pyPhotometry, 130 Hz)",
         "Wilbrecht NAc dLight (DANDI:001340, 30 Hz)"],
    )

    rec_file = ev_file = None
    if demo.startswith("—"):
        rec_file = st.file_uploader("Recording (.csv with signal + control)",
                                    type=["csv", "tsv", "txt"])
        ev_file = st.file_uploader("Event times (.csv or .npy, seconds)",
                                   type=["csv", "npy", "txt"])

    st.header("2 · The pipeline you actually ran")
    st.caption("So the tool can show you where your own analysis lands.")
    my_lp = st.selectbox("Low-pass cutoff (Hz)", [2.0, 4.0, 10.0], index=0)
    my_bl = st.selectbox("Bleaching correction", ["double_exp", "highpass"])
    my_mo = st.selectbox("Motion correction", ["OLS", "robust"])
    my_no = st.selectbox("Normalization", ["dFF", "zscore"])

    st.header("3 · What you are measuring")
    metric = st.selectbox("Downstream metric", ["mean", "peak", "auc"])
    pre_win = st.slider("Baseline window (s before event)", -5.0, 0.0, -1.0, 0.5)
    post_win = st.slider("Response window (s after event)", 0.5, 5.0, 2.0, 0.5)

    run = st.button("Run the multiverse", type="primary", width="stretch")


# ----------------------------------------------------------------- loading
# @st.cache_data(show_spinner=False)
def _load(kind, rec_bytes=None, ev_bytes=None, rec_name=None, ev_name=None):
    """Returns (Recording, list_of_warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "akam":
            rec = fqc.load("datasets/m53.ppd", events="datasets/m53_events.npy",
                           ppd_dir="../photometry_preprocessing")
        elif kind == "dandi":
            rec = fqc.load("datasets/nwb/sub-BSD011_ses-p148_rew.csv",
                           events="datasets/nwb/sub-BSD011_ses-p148_rew_events.csv")
        else:
            d = tempfile.mkdtemp()
            rp = os.path.join(d, rec_name)
            with open(rp, "wb") as f:
                f.write(rec_bytes)
            ep = None
            if ev_bytes is not None:
                ep = os.path.join(d, ev_name)
                with open(ep, "wb") as f:
                    f.write(ev_bytes)
            rec = fqc.load(rp, events=ep)
        # only surface warnings about the data, not Python housekeeping
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
    return rec, msgs




# ------------------------------------------------------------------ Claude
# The anthropic client sets up an httpx/SSL context that segfaults inside
# Streamlit's worker threads on macOS. Running it in a clean subprocess keeps
# it completely out of Streamlit's threading model.
import json
import subprocess
import sys


def _claude(kind, result, question=None):
    payload = {
        "kind": kind,
        "question": question,
        "rows": result.rows,
        "keys": result.keys,
        "name": result.recording.name,
        "metric": result.metric if isinstance(result.metric, str) else "custom",
        "baseline": list(result.baseline),
        "response": list(result.response),
        "my_pipeline": result.my_pipeline,
    }
    code = r"""
import json, sys
import numpy as np
import fiberqc as fqc
from fiberqc.multiverse import MultiverseResult

p = json.load(sys.stdin)
rec = fqc.Recording(signal=np.zeros(2), control=np.zeros(2),
                    time_s=np.arange(2, dtype=float), fs=1.0,
                    events=None, name=p["name"])
res = MultiverseResult(p["rows"], p["keys"], rec,
                       my_pipeline=p["my_pipeline"], metric=p["metric"],
                       baseline=tuple(p["baseline"]),
                       response=tuple(p["response"]))
out = res.ask(p["question"]) if p["kind"] == "ask" else res.report()
sys.stdout.write(out)
"""
    proc = subprocess.run([sys.executable, "-c", code],
                          input=json.dumps(payload), capture_output=True,
                          text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1]
                           if proc.stderr.strip() else "Claude call failed")
    return proc.stdout


rec, load_warnings = None, []
if run:
    try:
        if demo.startswith("Akam"):
            rec, load_warnings = _load("akam")
        elif demo.startswith("Wilbrecht"):
            rec, load_warnings = _load("dandi")
        elif rec_file is not None:
            rec, load_warnings = _load("upload", rec_file.getvalue(),
                                       ev_file.getvalue() if ev_file else None,
                                       rec_file.name,
                                       ev_file.name if ev_file else None)
        else:
            st.warning("Upload a recording, or pick one of the bundled examples.")
    except Exception as e:
        st.error(f"Could not load that recording: {e}")

# ----------------------------------------------------------------- run
if rec is not None:
    for m in load_warnings:
        st.warning(m, icon="⚠️")

    my_pipeline = {"low_pass": my_lp, "bleaching": my_bl,
                   "motion": my_mo, "normalization": my_no}

    with st.spinner(f"Running every reasonable pipeline on {rec.name}…"):
        result = fqc.multiverse(rec, my_pipeline=my_pipeline, metric=metric,
                                baseline=(pre_win, 0.0), response=(0.0, post_win))
    st.session_state["result"] = result

result = st.session_state.get("result")

if result is None:
    st.info("Pick a recording on the left, tell the tool which pipeline you ran, "
            "and press **Run the multiverse**.")
    st.stop()

stand = result.where_do_i_stand()
culprit, spread = result.culprit

# ----------------------------------------------------------------- verdict
if result.verdict == "FLIP":
    cls, head = "flip", "Your result depends on a preprocessing choice"
    if stand and stand["opposite_sign"]:
        body = (f"{stand['opposite_sign']} of {stand['n_alternatives']} reasonable "
                f"alternatives <b>reverse the direction</b> of your effect. "
                f"The choice that decides it is <b>{culprit}</b>.")
    else:
        body = (f"Only {result.n_significant} of {result.n} pipelines find the effect. "
                f"The choice that decides it is <b>{culprit}</b>.")
elif result.verdict == "ROBUST":
    cls, head = "robust", "Your result survives every reasonable pipeline"
    body = (f"All {result.n} specifications agree, in the same direction. "
            f"You can state this in front of a reviewer.")
else:
    cls, head = "null", "No pipeline finds an effect"
    body = "None of the specifications reach significance."

st.markdown(f'<div class="verdict {cls}"><h3>{head}</h3><p>{body}</p></div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
if stand:
    c1.markdown(f'<div class="statlab">your effect</div>'
                f'<div class="stat">d = {stand["my_d"]:.2f}</div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="statlab">agree with you</div>'
                f'<div class="stat">{stand["agree"]}/{stand["n_alternatives"]}</div>',
                unsafe_allow_html=True)
    alarm = " alarm" if stand["opposite_sign"] else ""
    c3.markdown(f'<div class="statlab">reverse your result</div>'
                f'<div class="stat{alarm}">{stand["opposite_sign"]}</div>',
                unsafe_allow_html=True)
lo, hi = result.d_range
c4.markdown(f'<div class="statlab">d across pipelines</div>'
            f'<div class="stat">{lo:.2f} … {hi:.2f}</div>',
            unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------- curve
tab_curve, tab_table, tab_ask = st.tabs(
    ["Specification curve", "Every pipeline", "Ask about your data"])

with tab_curve:
    path = os.path.join(tempfile.mkdtemp(), "spec.png")
    result.spec_curve(path)
    st.image(path, width="stretch")
    st.caption("Each dot is one complete analysis. The effect size leads; the "
               "t-statistic is shown as the evidence for it, not as the effect. "
               "The highlighted axis is the choice your conclusion hangs on.")
    with open(path, "rb") as f:
        st.download_button("Download figure (PNG)", f, "specification_curve.png",
                           "image/png")

with tab_table:
    df = result.to_df()
    shown = df.rename(columns={"d": "Cohen's d", "sig": "significant",
                               "mine": "your pipeline"})
    st.dataframe(
        shown, width="stretch", hide_index=True,
        column_config={
            "effect": st.column_config.NumberColumn("effect", format="%.4f"),
            "Cohen's d": st.column_config.NumberColumn("Cohen's d", format="%.2f"),
            "t": st.column_config.NumberColumn("t", format="%.2f"),
            "p": st.column_config.NumberColumn("p", format="%.1e"),
            "n_events": st.column_config.NumberColumn("events", format="%d"),
        },
    )
    st.caption("Cohen's d is the magnitude; t is the evidence for it. "
               "t = d × √n, so t grows with the number of events while d does not.")
    st.download_button("Download results (CSV)", df.to_csv(index=False),
                       "multiverse_results.csv", "text/csv")

with tab_ask:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("Set ANTHROPIC_API_KEY to ask questions about your result.")
    else:
        st.caption("Every number comes from the analysis above — Claude writes the "
                   "interpretation, never the statistics.")
        for role, msg in st.session_state.get("chat", []):
            with st.chat_message(role):
                st.markdown(msg)

        q = st.chat_input("e.g. Can I publish this? Which choice should I defend?")
        if q:
            st.session_state.setdefault("chat", []).append(("user", q))
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        a = _claude("ask", result, q)
                    except Exception as e:
                        a = f"Could not reach Claude: {e}"
                st.markdown(a)
            st.session_state["chat"].append(("assistant", a))

        if st.button("Write the methods paragraph and reviewer statement"):
            with st.spinner("Writing…"):
                try:
                    st.markdown(_claude("report", result))
                except Exception as e:
                    st.error(f"Could not reach Claude: {e}")
