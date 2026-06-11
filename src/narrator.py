# ── what this script does ──────────────────────────────────────────────
# takes computed breathing features and segment analysis
# generates a structured plain-English clinical narrative
# entirely rule-based — no ML, no LLM, pure Python logic
# output reads like a clinical respiratory report
# ──────────────────────────────────────────────────────────────────────

def format_time(seconds):
    """converts seconds to mm:ss string"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def describe_rate_change(prev_rate, curr_rate, threshold=2.0):
    """describes how breathing rate changed between segments"""
    diff = curr_rate - prev_rate
    if abs(diff) < threshold:
        return "remained stable"
    elif diff > 0:
        return f"increased slightly to {curr_rate:.1f} bpm"
    else:
        return f"decreased slightly to {curr_rate:.1f} bpm"


def generate_opening(stats, patient_id):
    """generates the opening summary paragraph"""
    dur   = stats['duration_min']
    rate  = stats['mean_rate']
    cv    = stats['cv_percent']
    n     = stats['n_breaths']
    rdesc = stats['rate_desc']
    regdesc = stats['reg_desc']

    lines = []
    lines.append(f"BREATHING PATTERN NARRATIVE REPORT")
    lines.append(f"Patient: {patient_id.upper()}")
    lines.append(f"Recording duration: {dur:.1f} minutes")
    lines.append(f"Signal source: PPG-derived respiratory signal")
    lines.append(f"Method: Bandpass filtration (0.1–0.5 Hz) + peak detection")
    lines.append("=" * 60)
    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)

    summary = (
        f"Over the {dur:.0f}-minute recording, the patient completed "
        f"{n} detected breaths at a mean rate of {rate:.1f} breaths per "
        f"minute. This rate is {rdesc}. Breathing rhythm was {regdesc}, "
        f"with a coefficient of variation of {cv:.1f}% across inter-breath "
        f"intervals."
    )
    lines.append(summary)
    lines.append("")

    # clinical interpretation of overall rate
    if stats['rate_category'] == 'normal':
        lines.append(
            "Respiratory rate was within the normal adult range "
            "(12–20 breaths/min) throughout the recording."
        )
    elif stats['rate_category'] == 'elevated':
        lines.append(
            "Respiratory rate was mildly elevated above the normal adult "
            "range (12–20 breaths/min). This may reflect physiological "
            "stress, pain, anxiety, or underlying respiratory demand. "
            "Clinical correlation is advised."
        )
    elif stats['rate_category'] == 'tachypnea':
        lines.append(
            "CLINICAL NOTE: Respiratory rate exceeded 25 breaths/min, "
            "meeting criteria for tachypnea. This warrants clinical "
            "evaluation for underlying causes including respiratory "
            "distress, infection, or metabolic acidosis."
        )
    elif stats['rate_category'] in ('slow', 'bradypnea'):
        lines.append(
            "CLINICAL NOTE: Respiratory rate was below normal range. "
            "Bradypnea may indicate sedation effects, neurological "
            "depression, or metabolic alkalosis. Clinical review recommended."
        )

    return lines


def generate_segment_narrative(segments):
    """generates paragraph for each time segment"""
    lines = []
    lines.append("")
    lines.append("SEGMENT-BY-SEGMENT ANALYSIS")
    lines.append("-" * 40)

    for i, seg in enumerate(segments):
        start = format_time(seg['start_sec'])
        end   = format_time(seg['end_sec'])
        rate  = seg['rate_bpm']
        cv    = seg['cv_percent']
        n     = seg['n_breaths']
        rdesc = seg['rate_desc']
        regdesc = seg['reg_desc']

        lines.append(f"\nSegment {seg['segment']} "
                     f"[{start} – {end}]")

        # rate description
        if i == 0:
            rate_sentence = (
                f"Breathing rate was {rate:.1f} breaths/min "
                f"({rdesc})."
            )
        else:
            prev_rate = segments[i-1]['rate_bpm']
            change    = describe_rate_change(prev_rate, rate)
            rate_sentence = (
                f"Breathing rate {change} "
                f"({rate:.1f} bpm, {rdesc})."
            )
        lines.append(rate_sentence)

        # regularity description
        if seg['reg_category'] == 'regular':
            reg_sentence = (
                f"Rhythm was consistent (CV {cv:.1f}%), "
                f"indicating stable respiratory control."
            )
        elif seg['reg_category'] == 'variable':
            reg_sentence = (
                f"Rhythm showed moderate variability (CV {cv:.1f}%), "
                f"which may reflect normal physiological variation "
                f"or mild respiratory irregularity."
            )
        else:
            reg_sentence = (
                f"Rhythm was highly irregular (CV {cv:.1f}%), "
                f"suggesting possible respiratory disturbance or "
                f"patient movement artefact during this period."
            )
        lines.append(reg_sentence)

        # breath count context
        lines.append(
            f"{n} breaths detected in this 2-minute window."
        )

    return lines


def generate_episode_narrative(episodes):
    """generates episode alerts if any abnormal periods detected"""
    lines = []
    lines.append("")
    lines.append("EPISODE ALERTS")
    lines.append("-" * 40)

    if not episodes:
        lines.append(
            "No sustained episodes of tachypnea (>25 bpm) or "
            "bradypnea (<10 bpm) were detected during this recording. "
            "Breathing rate remained within or near normal limits throughout."
        )
        return lines

    for ep in episodes:
        start = format_time(ep['start'])
        end   = format_time(ep['end'])
        dur   = ep['duration']

        if ep['type'] == 'tachypnea':
            lines.append(
                f"ELEVATED RATE EPISODE [{start}–{end}]: "
                f"Breathing rate exceeded 25 bpm for {dur:.0f} seconds "
                f"(mean {ep['mean_rate']:.1f} bpm, "
                f"peak {ep['max_rate']:.1f} bpm). "
                f"Sustained tachypnea of this duration warrants "
                f"clinical review."
            )
        else:
            lines.append(
                f"SLOW RATE EPISODE [{start}–{end}]: "
                f"Breathing rate fell below 10 bpm for {dur:.0f} seconds "
                f"(mean {ep['mean_rate']:.1f} bpm). "
                f"Bradypnea of this duration should be clinically evaluated."
            )

    return lines


def generate_ibi_narrative(stats):
    """generates inter-breath interval analysis paragraph"""
    lines = []
    lines.append("")
    lines.append("BREATHING INTERVAL ANALYSIS")
    lines.append("-" * 40)

    mean_ibi = stats['mean_ibi']
    std_ibi  = stats['std_ibi']
    min_ibi  = stats['min_ibi']
    max_ibi  = stats['max_ibi']
    cv       = stats['cv_percent']

    lines.append(
        f"Mean inter-breath interval: {mean_ibi:.2f} seconds "
        f"(± {std_ibi:.2f}s SD)."
    )
    lines.append(
        f"Range: {min_ibi:.2f}s (shortest breath) to "
        f"{max_ibi:.2f}s (longest breath)."
    )

    spread = max_ibi - min_ibi
    if spread > 3.0:
        lines.append(
            f"The {spread:.1f}s spread between shortest and longest "
            f"breath intervals indicates notable variability in "
            f"respiratory effort across the recording."
        )
    else:
        lines.append(
            f"The {spread:.1f}s spread between shortest and longest "
            f"breath intervals indicates relatively uniform "
            f"respiratory effort."
        )

    return lines


def generate_closing(stats, correlation):
    """generates closing methodology and limitations note"""
    lines = []
    lines.append("")
    lines.append("METHODOLOGY AND LIMITATIONS")
    lines.append("-" * 40)
    lines.append(
        "Breathing signal was derived from photoplethysmography (PPG) "
        "using respiratory modulation of blood volume — a slow oscillation "
        "(0.1–0.5 Hz) superimposed on the cardiac PPG waveform. This was "
        "isolated using a zero-phase Butterworth bandpass filter. "
        "Individual breaths were detected via prominence-filtered "
        "peak detection with a minimum inter-breath interval constraint."
    )
    lines.append("")
    lines.append(
        f"Validation: PPG-derived breathing signal achieved a Pearson "
        f"correlation of {abs(correlation):.3f} with the reference "
        f"chest sensor signal."
    )
    lines.append("")
    lines.append(
        "Limitations: PPG-derived respiratory rate may be affected by "
        "patient movement, poor perfusion, or irregular cardiac rhythm. "
        "This report is generated for research purposes only and does "
        "not constitute clinical diagnosis."
    )
    return lines


def generate_narrative(stats, segments, episodes,
                       correlation, patient_id):
    """assembles the full narrative report"""
    lines = []
    lines += generate_opening(stats, patient_id)
    lines += generate_segment_narrative(segments)
    lines += generate_episode_narrative(episodes)
    lines += generate_ibi_narrative(stats)
    lines += generate_closing(stats, correlation)
    return "\n".join(lines)


# ── test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.insert(0, 'src')
    from load_data import load_record, extract_signals
    from dsp       import (extract_breathing_from_ppg,
                           detect_breaths,
                           validate_against_reference)
    from features  import (segment_analysis,
                           compute_overall_stats,
                           detect_episodes)

    PATIENT = 'bidmc01'
    record               = load_record(PATIENT)
    ppg, resp, hr, t, fs = extract_signals(record)
    breathing            = extract_breathing_from_ppg(ppg, fs, resp)
    breath_peaks         = detect_breaths(breathing, fs)
    correlation          = validate_against_reference(
                               breathing, resp, fs, PATIENT)

    stats    = compute_overall_stats(breath_peaks, fs, len(ppg))
    segments = segment_analysis(breath_peaks, fs, len(ppg))

    rate_times  = np.array([s['start_sec'] for s in segments])
    rate_values = np.array([s['rate_bpm']  for s in segments])
    episodes    = detect_episodes(rate_times, rate_values)

    narrative = generate_narrative(
        stats, segments, episodes, correlation, PATIENT)

    print(narrative)

    # save to file
    with open(f'output/{PATIENT}_narrative.txt', 'w') as f:
        f.write(narrative)
    print(f"\nNarrative saved to output/{PATIENT}_narrative.txt")
    