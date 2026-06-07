import streamlit as st
import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt

NS = {"hl7": "urn:hl7-org:v3"}

LEAD_LABELS = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL",
    "MDC_ECG_LEAD_AVF": "aVF",
    "MDC_ECG_LEAD_V": "V",
}


def bandpass(sig, fs):
    b, a = butter(3, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def _rpeak_output(peaks, dt):
    peak_times = peaks * dt
    n_beats = len(peaks)
    if n_beats >= 2:
        rr = np.diff(peaks) * dt
        hr = 60.0 / np.median(rr)
    else:
        rr = np.array([])
        hr = None
    return peaks, peak_times, n_beats, rr, hr


def compute_qrs_duration(sig, dt, r_peaks):
    fs = 1.0 / dt
    work = savgol_filter(sig, 11, 2)
    onsets = []
    offsets = []
    for rp in r_peaks:
        win_start = max(0, rp - int(0.10 * fs))
        pre = work[win_start:rp + 1]
        if len(pre) < 4:
            onsets.append(rp)
            offsets.append(rp)
            continue
        d = np.diff(pre)
        d_max = float(np.max(np.abs(d)))
        if d_max == 0:
            onsets.append(rp)
            offsets.append(rp)
            continue
        q_idx = win_start + int(np.argmin(pre))
        thresh = 0.02 * d_max
        ons = q_idx
        for i in range(q_idx, win_start, -1):
            if abs(work[i] - work[i - 1]) < thresh:
                ons = i
                break
        if ons == q_idx:
            ons = win_start

        win_end = min(len(sig), rp + int(0.12 * fs))
        post = work[rp:win_end + 1]
        if len(post) < 4:
            onsets.append(max(ons, 0))
            offsets.append(rp)
            continue
        d = np.diff(post)
        d_max = float(np.max(np.abs(d)))
        if d_max == 0:
            onsets.append(max(ons, 0))
            offsets.append(rp)
            continue
        s_search = min(len(post), int(0.08 * fs))
        s_local = int(np.argmin(post[:s_search]))
        s_idx = rp + s_local
        thresh = 0.02 * d_max
        off = s_idx
        for i in range(s_local, len(d)):
            if abs(work[rp + i] - work[rp + i - 1]) < thresh:
                off = rp + i
                break
        if off == s_idx:
            off = win_end

        onsets.append(max(ons, 0))
        offsets.append(min(off, len(sig) - 1))
    onsets = np.array(onsets, dtype=int)
    offsets = np.array(offsets, dtype=int)
    durations_ms = (offsets - onsets) * dt * 1000.0
    return onsets, offsets, durations_ms


def compute_pr_interval(sig, dt, r_peaks, qrs_onsets):
    sig = np.asarray(sig)
    fs = 1.0 / dt
    work = savgol_filter(sig, 11, 2)
    pr_vals = []
    sig_rms = float(np.sqrt(np.mean(sig ** 2)))
    for rp, qrs_on in zip(r_peaks, qrs_onsets):
        win_end = max(0, qrs_on - int(0.02 * fs))
        win_start = max(0, qrs_on - int(0.22 * fs))
        segment = work[win_start:win_end + 1]
        if len(segment) < 8:
            pr_vals.append(np.nan)
            continue
        baseline = float(np.median(segment[:max(1, len(segment) // 4)]))
        p_local = int(np.argmax(np.abs(segment - baseline)))
        p_peak = win_start + p_local
        p_amp = abs(segment[p_local] - baseline)
        noise_std = max(1e-8, float(np.std(segment[:max(1, len(segment) // 4)])))
        if p_amp < 2.5 * noise_std or p_amp > 0.75 * sig_rms:
            pr_vals.append(np.nan)
            continue
        if p_peak >= win_end:
            pr_vals.append(np.nan)
            continue
        p_ons = p_peak
        for i in range(p_peak, win_start, -1):
            if abs(work[i] - baseline) < 0.15 * p_amp:
                p_ons = i
                break
        pr = (qrs_on - p_ons) * dt * 1000.0
        pr_vals.append(pr if 60 <= pr <= 280 else np.nan)
    return np.array(pr_vals)


def _robust_mean(values):
    valid = values[~np.isnan(values)]
    if len(valid) < 1:
        return np.nan
    if len(valid) < 4:
        return float(np.mean(valid))
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad < 1e-8:
        return float(median)
    threshold = 2.0 * 1.4826 * mad
    cleaned = valid[np.abs(valid - median) < threshold]
    if len(cleaned) < 2:
        cleaned = valid
    return float(np.mean(cleaned))


def compute_qt_interval(sig, dt, r_peaks, qrs_onsets):
    fs = 1.0 / dt
    work = savgol_filter(sig, 11, 2)
    sig_max = float(np.max(np.abs(work)))
    if sig_max < 1e-8:
        return np.full(len(r_peaks), np.nan), np.full(len(r_peaks), np.nan)
    qt_vals = []
    t_ends = []
    for i, (rp, qrs_on) in enumerate(zip(r_peaks, qrs_onsets)):
        t_start = qrs_on + int(0.12 * fs)
        t_end_max = qrs_on + int(0.48 * fs)
        if i < len(r_peaks) - 1:
            t_end = min(qrs_onsets[i + 1] - int(0.04 * fs), t_end_max)
        else:
            t_end = min(len(sig), t_end_max)
        t_end = min(t_end, len(sig))
        if t_start >= t_end:
            qt_vals.append(np.nan)
            t_ends.append(qrs_on)
            continue
        segment = work[t_start:t_end]
        if len(segment) < 5:
            qt_vals.append(np.nan)
            t_ends.append(qrs_on)
            continue
        baseline = float(np.median(segment))
        qrs_left = max(0, rp - int(0.04 * fs))
        qrs_right = min(len(work), rp + int(0.04 * fs))
        qrs_win = work[qrs_left:qrs_right]
        if len(qrs_win) < 3:
            qt_vals.append(np.nan)
            t_ends.append(qrs_on)
            continue
        qrs_polarity = 1.0 if (float(np.max(qrs_win)) - baseline) >= abs(float(np.min(qrs_win)) - baseline) else -1.0
        peak_start = max(t_start, rp + int(0.15 * fs))
        peak_end = min(t_end, rp + int(0.40 * fs))
        if peak_start >= peak_end:
            qt_vals.append(np.nan)
            t_ends.append(qrs_on)
            continue
        peak_segment = work[peak_start:peak_end]
        if qrs_polarity > 0:
            t_peak_local = int(np.argmax(peak_segment))
        else:
            t_peak_local = int(np.argmin(peak_segment))
        t_peak_idx = peak_start + t_peak_local
        t_amp = abs(work[t_peak_idx] - baseline)
        if t_amp < max(0.01 * sig_max, 15.0):
            qt_vals.append(np.nan)
            t_ends.append(qrs_on)
            continue
        noise_region = work[t_start:peak_start]
        if len(noise_region) > 5:
            x = np.arange(len(noise_region), dtype=float)
            if len(noise_region) > 2:
                coeffs = np.polyfit(x, noise_region, 1)
                trend = np.polyval(coeffs, x)
                noise_detrended = noise_region - trend
            else:
                noise_detrended = noise_region
            noise_std = max(float(np.std(noise_detrended)), 1.0)
        else:
            noise_std = max(float(np.std(segment)) * 0.5, 1.0)
        post = work[t_peak_idx:t_end]
        t_end_idx = t_peak_idx
        found = False
        end_threshold = max(0.10 * t_amp, 3.0 * noise_std)
        min_stable = int(0.03 * fs)
        for j in range(1, len(post)):
            if abs(post[j] - baseline) < end_threshold:
                la_end = min(j + min_stable, len(post))
                if all(abs(post[k] - baseline) < end_threshold * 1.5 for k in range(j, la_end)):
                    t_end_idx = t_peak_idx + j
                    found = True
                    break
        if not found:
            skip_biphasic = int(0.02 * fs)
            remaining = work[min(t_peak_idx + skip_biphasic, t_end):t_end]
            if len(remaining) >= 5:
                centered = remaining - baseline
                zero_crossings = np.where(np.diff(np.sign(centered)) != 0)[0]
                if len(zero_crossings) > 0:
                    search_start = t_peak_idx + skip_biphasic + int(zero_crossings[-1]) + 4
                    if search_start < t_end:
                        post2 = work[search_start:t_end]
                        if len(post2) >= 5:
                            for j in range(1, len(post2)):
                                if abs(post2[j] - baseline) < end_threshold:
                                    la_end = min(j + min_stable, len(post2))
                                    if all(abs(post2[k] - baseline) < end_threshold * 1.5 for k in range(j, la_end)):
                                        t_end_idx = search_start + j
                                        found = True
                                        break
        if not found and len(post) >= 10:
            d = np.diff(post)
            d_abs = np.abs(d)
            win_size = min(5, len(d_abs) - 1) if len(d_abs) % 2 == 0 else min(5, len(d_abs))
            if win_size >= 3 and len(d_abs) >= win_size:
                d_smooth = savgol_filter(d_abs, win_size, 1)
            else:
                d_smooth = d_abs
            d_thresh = 0.05 * float(np.max(d_smooth)) if np.max(d_smooth) > 0 else 0
            stable_count = 0
            for j in range(0, len(d_smooth)):
                if d_smooth[j] <= d_thresh:
                    stable_count += 1
                    if stable_count >= min_stable:
                        t_end_idx = t_peak_idx + j - min_stable + 1
                        found = True
                        break
                else:
                    stable_count = 0
        if not found:
            t_end_idx = t_peak_idx + int(0.08 * fs)
        t_end_idx = min(t_end_idx, t_end)
        qt_ms = (t_end_idx - qrs_on) * dt * 1000.0
        if qt_ms < 200.0 or qt_ms > 600.0:
            qt_ms = np.nan
        qt_vals.append(qt_ms)
        t_ends.append(t_end_idx)
    return np.array(qt_vals), np.array(t_ends)


def compute_qtc_bazett(qt_ms, rr_seconds):
    if rr_seconds is None or rr_seconds <= 0:
        return np.nan
    return qt_ms / np.sqrt(rr_seconds)


def compute_qrs_axis(leads, dt, ref_code="MDC_ECG_LEAD_II"):
    L1 = "MDC_ECG_LEAD_I"
    L2 = "MDC_ECG_LEAD_II"
    L3 = "MDC_ECG_LEAD_III"
    AVF = "MDC_ECG_LEAD_AVF"
    if L1 not in leads:
        return np.nan, "Lead I unavailable", np.nan, np.nan, 0.0
    ref = ref_code if ref_code in leads else next(iter(leads))
    sig_ref = np.array(leads[ref]["signal"])
    peaks, _, n_beats, _, _ = detect_r_peaks(sig_ref, dt)
    if n_beats < 2:
        return np.nan, "Insufficient beats", np.nan, np.nan, 0.0
    qrs_onsets, qrs_offsets, _ = compute_qrs_duration(sig_ref, dt, peaks)

    def _net_amp(code, onset, offset):
        if code not in leads:
            return np.nan
        sig = np.array(leads[code]["signal"])
        if offset >= len(sig) or onset < 0:
            return np.nan
        pre_start = max(0, onset - int(0.04 / dt))
        pre_end = max(0, onset - 1)
        if pre_end > pre_start + 2:
            baseline = float(np.median(sig[pre_start:pre_end]))
        else:
            baseline = 0.0
        qrs = sig[onset:offset + 1] - baseline
        if float(np.max(qrs) - np.min(qrs)) <= 30.0:
            return np.nan
        return float(np.mean(qrs))

    beat_i = []
    beat_avf = []
    for onset, offset in zip(qrs_onsets, qrs_offsets):
        if onset >= offset or onset < 0:
            continue
        ni = _net_amp(L1, onset, offset)
        if np.isnan(ni):
            continue
        if AVF in leads:
            nf = _net_amp(AVF, onset, offset)
        elif L2 in leads and L3 in leads:
            n2 = _net_amp(L2, onset, offset)
            n3 = _net_amp(L3, onset, offset)
            nf = (n2 + n3) / 2.0 if not (np.isnan(n2) or np.isnan(n3)) else np.nan
        else:
            nf = np.nan
        if not np.isnan(nf):
            beat_i.append(ni)
            beat_avf.append(nf)

    if len(beat_i) < 2:
        return np.nan, "No valid QRS in Lead I", np.nan, np.nan, 0.0

    ai = float(np.mean(beat_i))
    avf = float(np.mean(beat_avf))
    axis_deg = float(np.degrees(np.arctan2(avf, ai)))

    per_beat_axes = [float(np.degrees(np.arctan2(beat_avf[j], beat_i[j]))) for j in range(len(beat_i))]
    angles_rad = np.radians(per_beat_axes)
    mean_sin = float(np.mean(np.sin(angles_rad)))
    mean_cos = float(np.mean(np.cos(angles_rad)))
    r = min(np.sqrt(mean_sin ** 2 + mean_cos ** 2), 1.0)
    confidence = r * 100.0

    classification = _classify_qrs_axis(axis_deg)
    return axis_deg, classification, ai, avf, confidence


def _classify_qrs_axis(axis_deg):
    bounds = [(-90, "Left Axis Deviation", "Extreme Axis Deviation"),
              (-30, "Normal Axis", "Left Axis Deviation"),
              (90, "Normal Axis", "Right Axis Deviation"),
              (180, "Right Axis Deviation", "Extreme Axis Deviation")]
    for bound, left, right in bounds:
        if abs(axis_deg - bound) <= 5:
            return f"Borderline ({left}/{right})"
    if abs(axis_deg + 180) <= 5:
        return "Borderline (Right Axis Deviation/Extreme Axis Deviation)"
    if -30 <= axis_deg <= 90:
        return "Normal Axis"
    if -90 <= axis_deg < -30:
        return "Left Axis Deviation"
    if 90 < axis_deg <= 180:
        return "Right Axis Deviation"
    if -180 < axis_deg < -90:
        return "Extreme Axis Deviation"
    return "Extreme Axis Deviation" if abs(axis_deg) == 180 else "Indeterminate"


def _interpret_hr(bpm):
    if bpm is None or (isinstance(bpm, float) and np.isnan(bpm)):
        return "Unknown", "Heart rate could not be determined."
    if bpm < 55:
        return "Bradycardia", f"Heart rate of {bpm:.0f} bpm is below 55 bpm, indicating sinus bradycardia."
    if bpm < 65:
        return "Borderline (Bradycardia)", f"Heart rate of {bpm:.0f} bpm is near the bradycardia threshold (55–65 bpm). Clinical correlation advised."
    if bpm <= 95:
        return "Normal", f"Heart rate of {bpm:.0f} bpm is within the normal range (60–100 bpm)."
    if bpm <= 105:
        return "Borderline (Tachycardia)", f"Heart rate of {bpm:.0f} bpm is near the tachycardia threshold (95–105 bpm). Clinical correlation advised."
    return "Tachycardia", f"Heart rate of {bpm:.0f} bpm exceeds 105 bpm, indicating sinus tachycardia."


def _interpret_pr(pr_ms):
    if pr_ms is None or (isinstance(pr_ms, float) and np.isnan(pr_ms)):
        return "Unknown", "PR interval could not be measured reliably."
    if pr_ms < 110:
        return "Short PR", f"PR interval of {pr_ms:.0f} ms is shorter than 110 ms. Consider pre-excitation (e.g., WPW pattern)."
    if pr_ms < 130:
        return "Borderline (Short PR)", f"PR interval of {pr_ms:.0f} ms is near the short PR threshold (110–130 ms). Clinical correlation advised."
    if pr_ms <= 190:
        return "Normal", f"PR interval of {pr_ms:.0f} ms is within normal limits (120–200 ms)."
    if pr_ms <= 210:
        return "Borderline (Prolonged PR)", f"PR interval of {pr_ms:.0f} ms is near the prolonged PR threshold (190–210 ms). Clinical correlation advised."
    return "Prolonged PR", f"PR interval of {pr_ms:.0f} ms exceeds 210 ms, consistent with first-degree AV block."


def _interpret_qrs(qrs_ms):
    if qrs_ms is None or (isinstance(qrs_ms, float) and np.isnan(qrs_ms)):
        return "Unknown", "QRS duration could not be measured reliably."
    if qrs_ms < 115:
        return "Normal", f"QRS duration of {qrs_ms:.0f} ms is within normal limits (<120 ms)."
    if qrs_ms <= 125:
        return "Borderline", f"QRS duration of {qrs_ms:.0f} ms is borderline prolonged (115–125 ms). Clinical correlation advised."
    return "Prolonged QRS", f"QRS duration of {qrs_ms:.0f} ms exceeds 125 ms, indicating ventricular conduction delay."


def _interpret_qtc(qtc_ms):
    if qtc_ms is None or (isinstance(qtc_ms, float) and np.isnan(qtc_ms)):
        return "Unknown", "QTc (Bazett) could not be determined."
    if qtc_ms <= 440:
        return "Normal", f"QTc of {qtc_ms:.0f} ms (Bazett) is within normal limits."
    if qtc_ms <= 460:
        return "Borderline", f"QTc of {qtc_ms:.0f} ms (Bazett) is borderline prolonged (440–460 ms). Clinical correlation advised."
    return "Prolonged", f"QTc of {qtc_ms:.0f} ms (Bazett) exceeds 460 ms, indicating prolonged QT. Increased risk of arrhythmias."


def _interpret_axis_label(axis_deg, axis_class):
    if axis_deg is None or (isinstance(axis_deg, float) and np.isnan(axis_deg)):
        return "Unknown", "QRS axis could not be determined."
    if "Borderline" in axis_class:
        return axis_class, f"QRS axis at {axis_deg:.0f}° lies near a clinical boundary. Manual verification recommended."
    if "Extreme" in axis_class:
        return axis_class, f"QRS axis at {axis_deg:.0f}° shows extreme axis deviation. Consider ventricular rhythms or congenital heart disease."
    if "Left" in axis_class:
        return axis_class, f"QRS axis at {axis_deg:.0f}° shows left axis deviation. Consider left anterior fascicular block or inferior myocardial infarction."
    if "Right" in axis_class:
        return axis_class, f"QRS axis at {axis_deg:.0f}° shows right axis deviation. Consider right ventricular hypertrophy or lateral myocardial infarction."
    if "Normal" in axis_class:
        return axis_class, f"QRS axis at {axis_deg:.0f}° is within the normal range (−30° to +90°)."
    return axis_class, f"QRS axis measured at {axis_deg:.0f}°."


def build_report_text(params):
    hr = params.get("heart_rate")
    n_beats = params.get("n_beats")
    mean_rr = params.get("mean_rr_ms")
    pr = params.get("pr_ms")
    qrs = params.get("qrs_ms")
    qt = params.get("qt_ms")
    qtc = params.get("qtc_ms")
    axis_deg = params.get("axis_deg")
    axis_class = params.get("axis_class")
    conf = params.get("confidence")
    net_i = params.get("net_i")
    net_avf = params.get("net_avf")

    hr_class, hr_comment = _interpret_hr(hr)
    pr_class, pr_comment = _interpret_pr(pr)
    qrs_class, qrs_comment = _interpret_qrs(qrs)
    qtc_class, qtc_comment = _interpret_qtc(qtc)
    axis_label, axis_comment = _interpret_axis_label(axis_deg, axis_class)

    lines = []
    lines.append("## ECG Medical Report")
    lines.append("")
    lines.append("### 1. ECG Analysis Summary")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    if hr is not None and not np.isnan(hr):
        lines.append(f"| Heart Rate | {hr:.1f} bpm |")
    if n_beats is not None:
        lines.append(f"| Detected R-peaks | {n_beats} |")
    if mean_rr is not None and not np.isnan(mean_rr):
        lines.append(f"| Mean RR Interval | {mean_rr:.1f} ms |")
    if pr is not None and not np.isnan(pr):
        lines.append(f"| Mean PR Interval | {pr:.1f} ms |")
    if qrs is not None and not np.isnan(qrs):
        lines.append(f"| Mean QRS Duration | {qrs:.1f} ms |")
    if qt is not None and not np.isnan(qt):
        lines.append(f"| Mean QT Interval | {qt:.1f} ms |")
    if qtc is not None and not np.isnan(qtc):
        lines.append(f"| QTc (Bazett) | {qtc:.1f} ms |")
    if axis_deg is not None and not np.isnan(axis_deg):
        label = f"{axis_deg:.1f}° ({axis_class})"
        if net_i is not None and net_avf is not None and not (np.isnan(net_i) or np.isnan(net_avf)):
            label += f" — I={net_i:.0f} aVF={net_avf:.0f} µV"
        lines.append(f"| QRS Axis | {label} |")
    if conf is not None and not np.isnan(conf):
        lines.append(f"| Axis Confidence | {conf:.0f}% |")
    lines.append("")

    lines.append("### 2. Clinical Interpretation")
    lines.append("")
    lines.append("| Parameter | Classification |")
    lines.append("|-----------|---------------|")
    lines.append(f"| Heart Rate | {hr_class} |")
    lines.append(f"| PR Interval | {pr_class} |")
    lines.append(f"| QRS Duration | {qrs_class} |")
    lines.append(f"| QTc (Bazett) | {qtc_class} |")
    lines.append(f"| QRS Axis | {axis_label} |")
    lines.append("")

    lines.append("### 3. Automated Comments")
    lines.append("")
    if conf is not None and not np.isnan(conf) and conf < 70:
        lines.append(f"- ⚠️ **Low confidence:** Axis confidence is {conf:.0f}%. Measurements should be interpreted with caution.")
    if hr_comment:
        lines.append(f"- {hr_comment}")
    if pr_comment:
        lines.append(f"- {pr_comment}")
    if qrs_comment:
        lines.append(f"- {qrs_comment}")
    if qtc_comment:
        lines.append(f"- {qtc_comment}")
    if axis_comment:
        lines.append(f"- {axis_comment}")
    if all(c is None or np.isnan(c) if isinstance(c, float) else False for c in [pr, qrs, qt, qtc, axis_deg]):
        lines.append("- **Insufficient data:** Most parameters could not be computed. A full 12-lead recording is recommended.")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated automatically. This is not a clinical diagnosis — consult a qualified physician.*")

    return "\n".join(lines)


def _pdf_escape(text):
    from xml.sax.saxutils import escape
    return escape(text)


def build_report_pdf(params, output_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=18, spaceAfter=12, alignment=1)
    section_style = ParagraphStyle("SectionTitle", parent=styles["Heading2"], fontSize=14, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#2C3E50"))
    normal = ParagraphStyle("Normal", parent=styles["Normal"], fontSize=10, spaceAfter=4, leading=14)
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, spaceBefore=8, textColor=colors.HexColor("#7F8C8D"))

    hr = params.get("heart_rate")
    n_beats = params.get("n_beats")
    mean_rr = params.get("mean_rr_ms")
    pr = params.get("pr_ms")
    qrs = params.get("qrs_ms")
    qt = params.get("qt_ms")
    qtc = params.get("qtc_ms")
    axis_deg = params.get("axis_deg")
    axis_class = params.get("axis_class")
    conf = params.get("confidence")
    net_i = params.get("net_i")
    net_avf = params.get("net_avf")

    hr_class, hr_comment = _interpret_hr(hr)
    pr_class, pr_comment = _interpret_pr(pr)
    qrs_class, qrs_comment = _interpret_qrs(qrs)
    qtc_class, qtc_comment = _interpret_qtc(qtc)
    axis_label, axis_comment = _interpret_axis_label(axis_deg, axis_class)

    elements = []
    elements.append(Paragraph("ECG Medical Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=1, spaceAfter=8))
    elements.append(Spacer(1, 4 * mm))

    # Section 1
    elements.append(Paragraph("1. ECG Analysis Summary", section_style))
    summary_data = [
        ["Parameter", "Value"],
    ]
    if hr is not None and not np.isnan(hr):
        summary_data.append(["Heart Rate", f"{hr:.1f} bpm"])
    if n_beats is not None:
        summary_data.append(["Detected R-peaks", str(n_beats)])
    if mean_rr is not None and not np.isnan(mean_rr):
        summary_data.append(["Mean RR Interval", f"{mean_rr:.1f} ms"])
    if pr is not None and not np.isnan(pr):
        summary_data.append(["Mean PR Interval", f"{pr:.1f} ms"])
    if qrs is not None and not np.isnan(qrs):
        summary_data.append(["Mean QRS Duration", f"{qrs:.1f} ms"])
    if qt is not None and not np.isnan(qt):
        summary_data.append(["Mean QT Interval", f"{qt:.1f} ms"])
    if qtc is not None and not np.isnan(qtc):
        summary_data.append(["QTc (Bazett)", f"{qtc:.1f} ms"])
    if axis_deg is not None and not np.isnan(axis_deg):
        ax_val = f"{axis_deg:.1f}° ({axis_class})"
        summary_data.append(["QRS Axis", ax_val])
    if conf is not None and not np.isnan(conf):
        summary_data.append(["Axis Confidence", f"{conf:.0f}%"])

    if len(summary_data) > 1:
        col_w = [6 * cm, 10 * cm]
        t = Table(summary_data, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8F9FA"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
    elements.append(Spacer(1, 6 * mm))

    # Section 2
    elements.append(Paragraph("2. Clinical Interpretation", section_style))
    interp_data = [
        ["Parameter", "Classification"],
        ["Heart Rate", hr_class],
        ["PR Interval", pr_class],
        ["QRS Duration", qrs_class],
        ["QTc (Bazett)", qtc_class],
        ["QRS Axis", axis_label],
    ]
    t2 = Table(interp_data, colWidths=col_w)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8F9FA"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 6 * mm))

    # Section 3
    elements.append(Paragraph("3. Automated Comments", section_style))
    if conf is not None and not np.isnan(conf) and conf < 70:
        elements.append(Paragraph(f"⚠ <b>Low confidence:</b> Axis confidence is {conf:.0f}%. Measurements should be interpreted with caution.", normal))
    for comment in [hr_comment, pr_comment, qrs_comment, qtc_comment, axis_comment]:
        if comment:
            elements.append(Paragraph(f"• {comment}", normal))
    elements.append(Spacer(1, 8 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, spaceAfter=4))
    elements.append(Paragraph("Report generated automatically. This is not a clinical diagnosis — consult a qualified physician.", note_style))
    doc.build(elements)


def detect_r_peaks(signal, dt):
    fs = 1.0 / dt
    signal_smooth = savgol_filter(signal, 11, 2)
    signal_filt = bandpass(signal_smooth, fs)

    pos_energy = np.sum(signal_filt[signal_filt > 0] ** 2)
    neg_energy = np.sum(signal_filt[signal_filt < 0] ** 2)
    work_signal = -signal_filt if neg_energy > pos_energy else signal_filt

    min_distance = int(0.22 / dt)

    # --- Strategy 1: Envelope-based detection (energy domain) ---
    envelope_win = max(int(0.04 / dt), 1)
    kernel = np.ones(envelope_win) / envelope_win
    qrs_envelope = np.convolve(work_signal ** 2, kernel, mode="same")

    env_median = np.median(qrs_envelope)
    env_q90, env_q10 = np.percentile(qrs_envelope, [90, 10])
    env_height = env_median + 0.5 * (env_q90 - env_q10)

    env_peaks, _ = find_peaks(qrs_envelope, height=env_height)
    if len(env_peaks) < 2:
        env_peaks, _ = find_peaks(qrs_envelope, height=env_median + 0.25 * (env_q90 - env_q10))

    # --- Strategy 2: Signal-based prominence detection ---
    sig_median = np.median(work_signal)
    q90, q10 = np.percentile(work_signal, [90, 10])
    q75, q25 = np.percentile(work_signal, [75, 25])
    sig_height = sig_median + 0.3 * (q90 - q10)
    sig_prominence = max(0.3 * (q75 - q25), 0.02 * np.max(np.abs(work_signal)))
    sig_peaks, _ = find_peaks(work_signal, height=sig_height, prominence=sig_prominence)

    # Merge candidates from both strategies
    all_candidates = np.unique(np.concatenate([env_peaks, sig_peaks]))
    n_initial = len(all_candidates)
    if n_initial == 0:
        print("  detect_r_peaks: 0 candidates found")
        return np.array([]), np.array([]), 0, np.array([]), None

    # Refine to local maxima in work_signal
    refine_win = int(0.05 / dt)
    refined = []
    for p in all_candidates:
        left = max(0, p - refine_win)
        right = min(len(work_signal), p + refine_win)
        refined.append(left + np.argmax(work_signal[left:right]))
    peaks = np.unique(refined)

    # Merge clusters within 100ms
    cluster_win = int(0.10 / dt)
    clustered = []
    i = 0
    while i < len(peaks):
        j = i
        while j < len(peaks) - 1 and (peaks[j + 1] - peaks[j]) < cluster_win:
            j += 1
        cluster = peaks[i:j + 1]
        best = cluster[np.argmax(work_signal[cluster])]
        clustered.append(best)
        i = j + 1
    peaks = np.array(clustered)

    print(f"  detect_r_peaks: {n_initial} candidates -> {len(peaks)} after merge")
    if len(peaks) < 2:
        print(f"  detect_r_peaks: insufficient peaks (< 2)")
        return _rpeak_output(peaks, dt)

    # T-wave suppression: timing + amplitude ratio
    cleaned = [peaks[0]]
    for p in peaks[1:]:
        gap = (p - cleaned[-1]) * dt
        h_prev = work_signal[cleaned[-1]]
        h_curr = work_signal[p]
        if 0.15 <= gap <= 0.40 and h_curr < 0.5 * h_prev:
            continue
        cleaned.append(p)
    peaks = np.array(cleaned)

    if len(peaks) < 2:
        return _rpeak_output(peaks, dt)

    # Local maximum validation
    local_win = int(0.06 / dt)
    validated = []
    for p in peaks:
        left = max(0, p - local_win)
        right = min(len(work_signal), p + local_win)
        local_prom = np.max(work_signal[left:right]) - np.min(work_signal[left:right])
        if local_prom > 0 and (work_signal[p] - np.min(work_signal[left:right])) > 0.30 * local_prom:
            validated.append(p)
    peaks = np.array(validated)

    if len(peaks) < 2:
        return _rpeak_output(peaks, dt)

    # Peak width validation: reject wide (T-wave) and narrow (noise) peaks
    width_filtered = []
    for p in peaks:
        half = work_signal[p] / 2
        left = p
        while left > 0 and work_signal[left] > half:
            left -= 1
        right = p
        while right < len(work_signal) - 1 and work_signal[right] > half:
            right += 1
        width_ms = (right - left) / fs * 1000
        if 10 < width_ms < 150:
            width_filtered.append(p)
    if len(width_filtered) >= 2:
        peaks = np.array(width_filtered)

    if len(peaks) < 2:
        return _rpeak_output(peaks, dt)

    # Envelope energy validation
    peak_env_vals = qrs_envelope[peaks]
    env_thresh = np.percentile(peak_env_vals, 25) * 0.5
    valid = [p for p in peaks if qrs_envelope[p] > env_thresh]
    if len(valid) >= 2:
        peaks = np.array(valid)

    # Refractory enforcement
    if len(peaks) >= 2:
        final = [peaks[0]]
        for p in peaks[1:]:
            if (p - final[-1]) >= min_distance:
                final.append(p)
        peaks = np.array(final)

    n_beats = len(peaks)
    if n_beats >= 2:
        rr = np.diff(peaks) * dt
        hr = 60.0 / np.median(rr)
        print(f"  detect_r_peaks: {n_beats} peaks, indices={peaks.tolist()}, HR={hr:.1f} bpm")
    else:
        print(f"  detect_r_peaks: {n_beats} peaks, indices={peaks.tolist()}")

    return _rpeak_output(peaks, dt)


def parse_ecg_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    time_seq = root.find(".//hl7:sequence/hl7:code[@code='TIME_ABSOLUTE']/..", NS)
    if time_seq is None:
        raise ValueError("TIME_ABSOLUTE sequence not found in XML")
    value_elem = time_seq.find("hl7:value", NS)
    head = float(value_elem.find("hl7:head", NS).attrib["value"])
    increment = float(value_elem.find("hl7:increment", NS).attrib["value"])

    leads = {}
    for seq in root.findall(".//hl7:sequence", NS):
        code_elem = seq.find("hl7:code", NS)
        if code_elem is None:
            continue
        code = code_elem.attrib.get("code", "")
        if not code.startswith("MDC_ECG_LEAD"):
            continue
        value_elem = seq.find("hl7:value", NS)
        if value_elem is None:
            continue
        digits_elem = value_elem.find("hl7:digits", NS)
        if digits_elem is None or not digits_elem.text:
            continue
        values = [float(v) for v in digits_elem.text.strip().split()]
        t = [head + i * increment for i in range(len(values))]
        leads[code] = {"time": t, "signal": values}

    return leads, increment


st.set_page_config(
    page_title="ECG Digitizer Viewer",
    layout="wide",
)

st.title("ECG Digitizer Viewer")
st.markdown("Upload an ECG image, run ECGtizer, and view the digitized signal.")

uploaded_file = st.file_uploader(
    "Upload ECG Image", type=["png", "jpg", "jpeg", "pdf"]
)

input_path = None
output_path = None
temp_dir = None

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded ECG", use_container_width=True)

    if st.button("Run ECGtizer", type="primary"):
        with st.spinner("Running ECGtizer..."):
            try:
                temp_dir = tempfile.mkdtemp()
                ext = Path(uploaded_file.name).suffix
                input_path = os.path.join(temp_dir, f"input{ext}")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                output_path = os.path.join(temp_dir, "result.xml")

                cmd = [
                    "python",
                    "-m",
                    "ecgtizer.cli",
                    input_path,
                    "500",
                    "fragmented",
                    output_path,
                    "--verbose",
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                if result.returncode != 0:
                    st.error(f"ECGtizer failed:\n{result.stderr}")
                    st.stop()

                with st.expander("ECGtizer Output"):
                    st.text(result.stdout)
                    if result.stderr:
                        st.text(result.stderr)

                if not os.path.exists(output_path):
                    st.error("Output XML not found after execution.")
                    st.stop()

                leads, dt = parse_ecg_xml(output_path)

                if not leads:
                    st.warning("No leads found in the output XML.")
                    st.stop()

                st.success(
                    f"Successfully extracted {len(leads)} leads "
                    f"({len(next(iter(leads.values()))['signal'])} samples each, "
                    f"dt = {dt*1000:.2f} ms)"
                )

                n_leads = len(leads)
                fig, axes = plt.subplots(n_leads, 1, figsize=(12, 2.5 * n_leads), sharex=True)
                if n_leads == 1:
                    axes = [axes]

                for ax, (code, data) in zip(axes, leads.items()):
                    t = data["time"]
                    signal = data["signal"]
                    label = LEAD_LABELS.get(code, code)
                    ax.plot(t, signal, linewidth=0.8, color="black")
                    ax.set_ylabel(label, fontsize=12, fontweight="bold")
                    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
                    ax.margins(x=0.005)
                    ax.tick_params(axis="both", labelsize=9)
                    peaks, _, _, _, _ = detect_r_peaks(signal, dt)
                    ax.plot([t[i] for i in peaks], [signal[i] for i in peaks], "ro")

                axes[-1].set_xlabel("Time (s)", fontsize=11)
                fig.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                st.pyplot(fig)

                st.subheader("Peak Detection")
                lead_ii_code = "MDC_ECG_LEAD_II"
                params = None
                if lead_ii_code in leads:
                    signal = leads[lead_ii_code]["signal"]
                    peaks, peak_times, n_beats, rr, heart_rate = detect_r_peaks(signal, dt)
                    qrs_mean = None
                    pr_mean = None
                    qt_mean = None
                    qtc_mean = None
                    axis_deg = None
                    axis_class = None
                    net_i = None
                    net_avf = None
                    conf = None
                    if n_beats >= 2:
                        qrs_onsets, _, qrs_durs = compute_qrs_duration(signal, dt, peaks)
                        qrs_mean = float(np.mean(qrs_durs))
                        pr_vals = compute_pr_interval(signal, dt, peaks, qrs_onsets)
                        valid_pr = pr_vals[~np.isnan(pr_vals)]
                        if len(valid_pr) > 0:
                            pr_mean = float(np.mean(valid_pr))
                        qt_vals, _ = compute_qt_interval(signal, dt, peaks, qrs_onsets)
                        qt_mean = _robust_mean(qt_vals)
                        if not np.isnan(qt_mean):
                            mean_rr = float(np.mean(rr))
                            qtc_mean = float(compute_qtc_bazett(qt_mean, mean_rr))
                        axis_deg, axis_class, net_i, net_avf, conf = compute_qrs_axis(leads, dt)
                    if heart_rate is not None:
                        msg = (
                            f"Number of R-peaks detected: **{n_beats}**"
                            f" — Heart Rate: **{heart_rate:.1f} bpm**"
                            f" — Mean RR: **{np.mean(rr)*1000:.1f} ms**"
                            f" — SDNN: **{np.std(rr)*1000:.1f} ms**"
                        )
                        if qrs_mean is not None:
                            msg += f" — Mean QRS: **{qrs_mean:.1f} ms**"
                        if pr_mean is not None:
                            msg += f" — Mean PR: **{pr_mean:.1f} ms**"
                        if qt_mean is not None and not np.isnan(qt_mean):
                            msg += f" — Mean QT: **{qt_mean:.1f} ms**"
                        if qtc_mean is not None and not np.isnan(qtc_mean):
                            msg += f" — QTc (Bazett): **{qtc_mean:.1f} ms**"
                        if axis_deg is not None and not np.isnan(axis_deg):
                            msg += f" — QRS Axis: **{axis_deg:.1f}°** ({axis_class})"
                            msg += f" — I={net_i:.0f} aVF={net_avf:.0f} µV"
                            msg += f" — Conf: **{conf:.0f}%**"
                        st.write(msg)
                    else:
                        st.write(f"Number of R-peaks detected: **{n_beats}** (insufficient for heart rate)")
                    st.caption("(Reference lead: II, polarity-aware detection, statistical adaptive threshold, QRS clustering, P/T suppression, local max validation, local energy validation, 220ms refractory period, adaptive T-wave end detection, QRS axis from I/aVF)")
                    if heart_rate is not None:
                        params = {
                            "heart_rate": heart_rate,
                            "n_beats": n_beats,
                            "mean_rr_ms": float(np.mean(rr) * 1000),
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
                else:
                    st.write("Lead II not available. Using first available lead as reference.")
                    code = next(iter(leads))
                    signal = leads[code]["signal"]
                    peaks, peak_times, n_beats, rr, heart_rate = detect_r_peaks(signal, dt)
                    label = LEAD_LABELS.get(code, code)
                    qrs_mean = None
                    pr_mean = None
                    qt_mean = None
                    qtc_mean = None
                    axis_deg = None
                    axis_class = None
                    net_i = None
                    net_avf = None
                    conf = None
                    if n_beats >= 2:
                        qrs_onsets, _, qrs_durs = compute_qrs_duration(signal, dt, peaks)
                        qrs_mean = float(np.mean(qrs_durs))
                        pr_vals = compute_pr_interval(signal, dt, peaks, qrs_onsets)
                        valid_pr = pr_vals[~np.isnan(pr_vals)]
                        if len(valid_pr) > 0:
                            pr_mean = float(np.mean(valid_pr))
                        qt_vals, _ = compute_qt_interval(signal, dt, peaks, qrs_onsets)
                        qt_mean = _robust_mean(qt_vals)
                        if not np.isnan(qt_mean):
                            mean_rr = float(np.mean(rr))
                            qtc_mean = float(compute_qtc_bazett(qt_mean, mean_rr))
                        axis_deg, axis_class, net_i, net_avf, conf = compute_qrs_axis(leads, dt, ref_code=code)
                    if heart_rate is not None:
                        msg = f"Number of R-peaks detected: **{n_beats}** (Lead {label}) — Heart Rate: **{heart_rate:.1f} bpm**"
                        if qrs_mean is not None:
                            msg += f" — Mean QRS: **{qrs_mean:.1f} ms**"
                        if pr_mean is not None:
                            msg += f" — Mean PR: **{pr_mean:.1f} ms**"
                        if qt_mean is not None and not np.isnan(qt_mean):
                            msg += f" — Mean QT: **{qt_mean:.1f} ms**"
                        if qtc_mean is not None and not np.isnan(qtc_mean):
                            msg += f" — QTc (Bazett): **{qtc_mean:.1f} ms**"
                        if axis_deg is not None and not np.isnan(axis_deg):
                            msg += f" — QRS Axis: **{axis_deg:.1f}°** ({axis_class})"
                            msg += f" — I={net_i:.0f} aVF={net_avf:.0f} µV"
                            msg += f" — Conf: **{conf:.0f}%**"
                        st.write(msg)
                        if heart_rate is not None:
                            params = {
                                "heart_rate": heart_rate,
                                "n_beats": n_beats,
                                "mean_rr_ms": float(np.mean(rr) * 1000),
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
                    else:
                        st.write(f"Number of R-peaks detected: **{n_beats}** (Lead {label}) (insufficient for heart rate)")

                with st.expander("Per-lead R-peak counts"):
                    for code, data in leads.items():
                        signal = data["signal"]
                        peaks, _, n_beats, _, _ = detect_r_peaks(signal, dt)
                        label = LEAD_LABELS.get(code, code)
                        if n_beats >= 2:
                            qrs_onsets, _, qrs_durs = compute_qrs_duration(signal, dt, peaks)
                            qrs_mean = float(np.mean(qrs_durs))
                            pr_vals = compute_pr_interval(signal, dt, peaks, qrs_onsets)
                            valid_pr = pr_vals[~np.isnan(pr_vals)]
                            if len(valid_pr) > 0:
                                pr_mean = float(np.mean(valid_pr))
                            qt_vals, _ = compute_qt_interval(signal, dt, peaks, qrs_onsets)
                            qt_mean = _robust_mean(qt_vals)
                            parts = [f"Lead {label}: {n_beats} R-peaks, Mean QRS: {qrs_mean:.1f} ms"]
                            if len(valid_pr) > 0:
                                parts.append(f"Mean PR: {pr_mean:.1f} ms")
                            if not np.isnan(qt_mean):
                                parts.append(f"Mean QT: {qt_mean:.1f} ms")
                            st.write(" — ".join(parts))
                        else:
                            st.write(f"Lead {label}: {n_beats} R-peaks")

                st.markdown("---")
                st.subheader("Manual Heart Rate Calculator")

                auto_failed = heart_rate is None
                if auto_failed:
                    st.warning(
                        "Automatic R-peak detection could not compute heart rate. "
                        "Use the manual calculator below."
                    )

                use_manual = st.checkbox(
                    "Calculate heart rate manually from grid squares (HR = 300 / large squares)",
                    value=auto_failed,
                    help="Count the number of large (5 mm) squares between two consecutive "
                         "R-peaks on standard ECG paper (25 mm/s). "
                         "Formula: Heart Rate = 300 / Number of Large Squares"
                )

                if use_manual:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        large_squares = st.number_input(
                            "Number of large squares between R-R intervals",
                            min_value=0.5,
                            max_value=50.0,
                            value=5.0,
                            step=0.5,
                            format="%.1f",
                            help="On standard ECG paper at 25 mm/s, "
                                 "1 large square = 5 mm = 0.2 seconds. "
                                 "Count squares between consecutive R-peak peaks."
                        )
                    with col2:
                        if large_squares > 0:
                            manual_hr = 300.0 / large_squares
                            st.metric(
                                label="Heart Rate (Manual)",
                                value=f"{manual_hr:.1f} bpm",
                            )
                            st.caption(
                                f"RR interval \u2248 {large_squares:.1f} \u00d7 0.2 s = "
                                f"{large_squares * 0.2:.2f} s"
                            )

                st.markdown("---")
                st.subheader("Medical Report")
                if params is not None and params.get("heart_rate") is not None and not np.isnan(params["heart_rate"]):
                    report_md = build_report_text(params)
                    st.markdown(report_md)
                    col_left, col_right = st.columns([1, 4])
                    with col_left:
                        pdf_buf = BytesIO()
                        build_report_pdf(params, pdf_buf)
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_buf.getvalue(),
                            file_name="ecg_medical_report.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.info("Complete an ECG analysis with at least 2 detected beats to generate a medical report.")

            except subprocess.TimeoutExpired:
                st.error("ECGtizer timed out (5 minutes). The image may be too complex.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
