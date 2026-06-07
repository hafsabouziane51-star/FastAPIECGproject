import sys
import types
import io
import os
import base64
import tempfile
import subprocess
import shutil
from pathlib import Path
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




_mock_st = types.ModuleType("streamlit")

def _noop(*args, **kwargs): pass
def _noop_enter_exit(*args, **kwargs):
    class _Ctx:
        def __enter__(self2): return self2
        def __exit__(self2, *a): pass
    return _Ctx()
def _noop_cols(*args, **kwargs):
    class _Col:
        def __enter__(self2): return self2
        def __exit__(self2, *a): pass
        def __call__(self2, *a, **kw): return self2
    return (_Col(), _Col())

_mock_st.set_page_config = _noop
_mock_st.title = _noop
_mock_st.markdown = _noop
_mock_st.file_uploader = lambda *a, **kw: None
_mock_st.button = lambda *a, **kw: False
_mock_st.spinner = _noop_enter_exit
_mock_st.error = _noop
_mock_st.success = _noop
_mock_st.warning = _noop
_mock_st.info = _noop
_mock_st.write = _noop
_mock_st.text = _noop
_mock_st.metric = _noop
_mock_st.columns = _noop_cols
_mock_st.expander = _noop_enter_exit
_mock_st.pyplot = _noop
_mock_st.stop = lambda: (_ for _ in ()).throw(EOFError)
_mock_st.checkbox = lambda *a, **kw: False
_mock_st.number_input = lambda *a, **kw: 5.0
_mock_st.download_button = _noop
_mock_st.subheader = _noop
_mock_st.caption = _noop
_mock_st.image = _noop
_mock_st.empty = _noop

sys.modules["streamlit"] = _mock_st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Now import from app.py — module-level Streamlit calls use our mock
from app import (
    detect_r_peaks, compute_qrs_duration, compute_pr_interval,
    _robust_mean, compute_qt_interval, compute_qtc_bazett,
    compute_qrs_axis, build_report_text, build_report_pdf,
    parse_ecg_xml, _interpret_hr, _interpret_pr, _interpret_qrs,
    _interpret_qtc, _interpret_axis_label, LEAD_LABELS,
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


def _safe_val(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 2)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def analyze_ecg(file_bytes, filename):
    ext = Path(filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".pdf"):
        return {"error": f"Unsupported file type: {ext}"}

    tmp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(tmp_dir, f"input{ext}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        output_xml = os.path.join(tmp_dir, "result.xml")
        cmd = [
            sys.executable, "-m", "ecgtizer.cli",
            input_path, "500", "fragmented",
            output_xml, "--verbose",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(BASE_DIR),
        )

        result = {}

        if proc.returncode != 0:
            result["error"] = f"ECGtizer failed: {proc.stderr}"
            result["ecgtizer_output"] = proc.stdout
            result["ecgtizer_stderr"] = proc.stderr
            return result

        if not os.path.exists(output_xml):
            result["error"] = "Output XML not found"
            return result

        leads, dt = parse_ecg_xml(output_xml)
        if not leads:
            result["error"] = "No leads found in XML"
            return result

        n_leads = len(leads)
        n_samples = len(next(iter(leads.values()))["signal"])
        dt_ms = round(dt * 1000.0, 2)

        result["leads"] = [{"code": c, "label": LEAD_LABELS.get(c, c)} for c in leads]
        result["n_leads"] = n_leads
        result["n_samples"] = n_samples
        result["dt_ms"] = dt_ms
        result["ecgtizer_output"] = proc.stdout
        result["ecgtizer_stderr"] = proc.stderr

        # Generate plot
        n_plot = len(leads)
        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.5 * n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]
        for ax, (code, data) in zip(axes, leads.items()):
            t = data["time"]
            sig = data["signal"]
            label = LEAD_LABELS.get(code, code)
            ax.plot(t, sig, linewidth=0.8, color="black")
            ax.set_ylabel(label, fontsize=12, fontweight="bold")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax.margins(x=0.005)
            ax.tick_params(axis="both", labelsize=9)
            pk, _, _, _, _ = detect_r_peaks(sig, dt)
            ax.plot([t[i] for i in pk], [sig[i] for i in pk], "ro")
        axes[-1].set_xlabel("Time (s)", fontsize=11)
        fig.tight_layout()
        plot_buf = io.BytesIO()
        fig.savefig(plot_buf, format="png", dpi=150)
        plot_buf.seek(0)
        plt.close(fig)
        result["plot_base64"] = base64.b64encode(plot_buf.getvalue()).decode()

        # Reference lead analysis
        lead_ii = "MDC_ECG_LEAD_II"
        ref_code = lead_ii if lead_ii in leads else next(iter(leads))
        signal = leads[ref_code]["signal"]
        ref_label = LEAD_LABELS.get(ref_code, ref_code)

        peaks, peak_times, n_beats, rr, heart_rate = detect_r_peaks(signal, dt)
        fallback = ref_code != lead_ii

        result["ref_lead"] = ref_label
        result["ref_code"] = ref_code
        result["fallback"] = fallback
        result["heart_rate"] = _safe_val(heart_rate)
        result["n_beats"] = n_beats
        result["mean_rr_ms"] = _safe_val(np.mean(rr) * 1000) if n_beats >= 2 else None
        result["sdnn_ms"] = _safe_val(np.std(rr) * 1000) if n_beats >= 2 else None

        qrs_mean = pr_mean = qt_mean = qtc_mean = None
        axis_deg = axis_class = None
        net_i = net_avf = conf = None

        if n_beats >= 2:
            qrs_onsets, _, qrs_durs = compute_qrs_duration(signal, dt, peaks)
            qrs_mean = _safe_val(np.mean(qrs_durs))

            pr_vals = compute_pr_interval(signal, dt, peaks, qrs_onsets)
            vp = pr_vals[~np.isnan(pr_vals)]
            if len(vp):
                pr_mean = _safe_val(np.mean(vp))

            qt_vals, _ = compute_qt_interval(signal, dt, peaks, qrs_onsets)
            qv = _robust_mean(qt_vals)
            if not np.isnan(qv):
                qt_mean = _safe_val(qv)
                qtc_mean = _safe_val(compute_qtc_bazett(qv, float(np.mean(rr))))

            ax_res = compute_qrs_axis(leads, dt, ref_code=ref_code)
            if not np.isnan(ax_res[0]):
                axis_deg = round(float(ax_res[0]), 1)
            axis_class = ax_res[1]
            if not np.isnan(ax_res[2]):
                net_i = round(float(ax_res[2]))
            if not np.isnan(ax_res[3]):
                net_avf = round(float(ax_res[3]))
            if not np.isnan(ax_res[4]):
                conf = round(float(ax_res[4]))

        result["axis_deg"] = axis_deg
        result["axis_class"] = axis_class
        result["net_i"] = net_i
        result["net_avf"] = net_avf
        result["confidence"] = conf

        # Per-lead
        per_lead = {}
        result["qrs_mean"] = qrs_mean
        result["pr_mean"] = pr_mean
        result["qt_mean"] = qt_mean
        result["qtc_mean"] = qtc_mean
        for code, data in leads.items():
            sig = data["signal"]
            pk, _, nb, _, _ = detect_r_peaks(sig, dt)
            label = LEAD_LABELS.get(code, code)
            entry = {"label": label, "n_beats": nb}
            if nb >= 2:
                qo, _, qd = compute_qrs_duration(sig, dt, pk)
                entry["qrs_ms"] = _safe_val(np.mean(qd))
                pv = compute_pr_interval(sig, dt, pk, qo)
                vp = pv[~np.isnan(pv)]
                if len(vp):
                    entry["pr_ms"] = _safe_val(np.mean(vp))
                qv, _ = compute_qt_interval(sig, dt, pk, qo)
                qtv = _robust_mean(qv)
                if not np.isnan(qtv):
                    entry["qt_ms"] = _safe_val(qtv)
            per_lead[label] = entry
        result["per_lead"] = per_lead

        # Report
        params = {
            "heart_rate": result["heart_rate"],
            "n_beats": n_beats,
            "mean_rr_ms": result["mean_rr_ms"],
            "pr_ms": pr_mean,
            "qrs_ms": qrs_mean,
            "qt_ms": qt_mean,
            "qtc_ms": qtc_mean,
            "axis_deg": axis_deg,
            "axis_class": axis_class,
            "net_i": net_i,
            "net_avf": net_avf,
            "confidence": conf,
        }

        has_data = params["heart_rate"] is not None and not (isinstance(params["heart_rate"], float) and np.isnan(params["heart_rate"]))
        if has_data:
            result["report_text"] = build_report_text(params)
            hr_cls, hr_cmt = _interpret_hr(params["heart_rate"])
            pr_cls, pr_cmt = _interpret_pr(params["pr_ms"])
            qrs_cls, qrs_cmt = _interpret_qrs(params["qrs_ms"])
            qtc_cls, qtc_cmt = _interpret_qtc(params["qtc_ms"])
            ax_lbl, ax_cmt = _interpret_axis_label(params["axis_deg"], params["axis_class"])
            result["interpretations"] = {
                "heart_rate": {"classification": hr_cls, "comment": hr_cmt},
                "pr": {"classification": pr_cls, "comment": pr_cmt},
                "qrs": {"classification": qrs_cls, "comment": qrs_cmt},
                "qtc": {"classification": qtc_cls, "comment": qtc_cmt},
                "axis": {"classification": ax_lbl, "comment": ax_cmt},
            }
            pdf_buf = io.BytesIO()
            try:
                build_report_pdf(params, pdf_buf)
                result["pdf_base64"] = base64.b64encode(pdf_buf.getvalue()).decode()
            except Exception:
                result["pdf_base64"] = None
        else:
            result["report_text"] = None
            result["interpretations"] = None
            result["pdf_base64"] = None

        return result

    except subprocess.TimeoutExpired:
        return {"error": "ECGtizer timed out (5 minutes)"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- FastAPI App ---

app = FastAPI(title="ECG Digitizer Viewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".pdf"):
        return JSONResponse(
            {"error": f"Unsupported file type: {ext}"}, status_code=400
        )
    file_bytes = await file.read()
    result = analyze_ecg(file_bytes, file.filename)
    status = 400 if "error" in result and result.get("error") else 200
    return JSONResponse(result, status_code=status)


class ReportParams(BaseModel):
    heart_rate: float | None = None
    n_beats: int | None = None
    mean_rr_ms: float | None = None
    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    qtc_ms: float | None = None
    axis_deg: float | None = None
    axis_class: str | None = None
    net_i: int | None = None
    net_avf: int | None = None
    confidence: int | None = None


@app.post("/api/report/pdf")
async def report_pdf(params: ReportParams):
    buf = io.BytesIO()
    try:
        build_report_pdf(params.model_dump(), buf)
        pdf_bytes = buf.getvalue()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=ecg_medical_report.pdf"},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
