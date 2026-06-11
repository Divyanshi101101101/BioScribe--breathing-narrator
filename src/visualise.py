import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import sys
sys.path.insert(0, 'src')
from load_data import load_record, extract_signals
from dsp       import (extract_breathing_from_ppg, detect_breaths,
                        compute_breathing_rate, validate_against_reference)
from features  import (segment_analysis, compute_overall_stats,
                        detect_episodes)
from narrator  import generate_narrative

# ── what this script does ──────────────────────────────────────────────
# assembles the complete final output in one figure:
# left side  — signal plots (PPG, extracted breathing, rate over time)
# right side — narrative text report
# this is the demo screen — everything visible at once
# ──────────────────────────────────────────────────────────────────────

PATIENT = 'bidmc01'


def run_full_pipeline(patient_id):
    """runs complete pipeline and returns all results"""
    record               = load_record(patient_id)
    ppg, resp, hr, t, fs = extract_signals(record)
    breathing            = extract_breathing_from_ppg(ppg, fs, resp)
    breath_peaks         = detect_breaths(breathing, fs)
    rate_times, rates    = compute_breathing_rate(
                               breath_peaks, fs,
                               window_sec=30, step_sec=10,
                               total_samples=len(ppg))
    correlation          = validate_against_reference(
                               breathing, resp, fs, patient_id)
    stats                = compute_overall_stats(breath_peaks, fs, len(ppg))
    segments             = segment_analysis(breath_peaks, fs, len(ppg))
    rate_times_seg       = np.array([s['start_sec'] for s in segments])
    rate_values_seg      = np.array([s['rate_bpm']  for s in segments])
    episodes             = detect_episodes(rate_times_seg, rate_values_seg)
    narrative            = generate_narrative(
                               stats, segments, episodes,
                               correlation, patient_id)

    return (ppg, resp, breathing, breath_peaks,
            rate_times, rates, t, fs,
            stats, segments, narrative)


def plot_full_report(patient_id):
    """creates the combined signal + narrative figure"""

    print(f"Running full pipeline for {patient_id}...")
    (ppg, resp, breathing, breath_peaks,
     rate_times, rates, t, fs,
     stats, segments, narrative) = run_full_pipeline(patient_id)

    # ── figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0F1117')

    # grid: 3 signal plots on left, narrative on right
    gs = gridspec.GridSpec(
        3, 2,
        figure     = fig,
        width_ratios = [1.4, 1],
        hspace     = 0.45,
        wspace     = 0.08,
        left       = 0.05,
        right      = 0.97,
        top        = 0.92,
        bottom     = 0.07
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1])   # narrative spans all 3 rows

    # common style
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#1C1D22')
        ax.tick_params(colors='#6B6A66', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2B30')

    # ── plot 1: raw PPG (first 60s) ───────────────────────────────────
    mask = t <= 60
    ax1.plot(t[mask], ppg[mask],
             color='#1D9E75', linewidth=0.6, alpha=0.9)
    ax1.set_xlim(0, 60)
    ax1.set_ylabel('Amplitude', color='#9B9A97', fontsize=8)
    ax1.set_title('Raw PPG signal — cardiac + respiratory components',
                  color='#E8E8E3', fontsize=9, pad=6)
    ax1.set_xlabel('Time (seconds)', color='#6B6A66', fontsize=8)

    # ── plot 2: extracted breathing vs reference ──────────────────────
    b60   = breathing[mask]
    r60   = resp[mask]
    b_norm = (b60 - b60.min()) / (b60.max() - b60.min() + 1e-9)
    r_norm = (r60 - r60.min()) / (r60.max() - r60.min() + 1e-9)

    # detected peaks in first 60s
    peaks_60 = breath_peaks[breath_peaks < 60 * fs]

    ax2.plot(t[mask], r_norm,
             color='#185FA5', linewidth=1.0, alpha=0.7,
             label='Reference (chest sensor)')
    ax2.plot(t[mask], b_norm,
             color='#E24B4A', linewidth=1.0, alpha=0.8,
             linestyle='--', label='Extracted from PPG')
    ax2.scatter(peaks_60 / fs, b_norm[peaks_60],
                color='white', s=20, zorder=5, alpha=0.8)
    ax2.set_xlim(0, 60)
    ax2.set_ylabel('Normalised', color='#9B9A97', fontsize=8)
    ax2.set_title(
        f'Breathing signal extracted via bandpass filter '
        f'— {len(peaks_60)} breaths in 60s',
        color='#E8E8E3', fontsize=9, pad=6)
    ax2.set_xlabel('Time (seconds)', color='#6B6A66', fontsize=8)
    ax2.legend(fontsize=7, loc='upper right',
               facecolor='#2A2B30', labelcolor='#9B9A97',
               edgecolor='none')

    # ── plot 3: breathing rate over time ──────────────────────────────
    ax3.plot(rate_times / 60, rates,
             color='#9B6FD4', linewidth=1.5)
    ax3.axhspan(12, 20, alpha=0.1, color='#1D9E75',
                label='Normal range (12–20 bpm)')
    ax3.axhline(20, color='#EF9F27', linewidth=0.8,
                linestyle='--', alpha=0.6)
    ax3.axhline(12, color='#EF9F27', linewidth=0.8,
                linestyle='--', alpha=0.6)

    # shade segments
    colours = ['#185FA5', '#1D9E75', '#9B6FD4', '#EF9F27']
    for i, seg in enumerate(segments):
        ax3.axvspan(seg['start_min'], seg['end_min'],
                    alpha=0.06,
                    color=colours[i % len(colours)])

    ax3.set_xlim(0, rate_times[-1] / 60 if len(rate_times) > 0 else 8)
    ax3.set_ylim(0, max(rates) * 1.3 if len(rates) > 0 else 30)
    ax3.set_ylabel('Breaths/min', color='#9B9A97', fontsize=8)
    ax3.set_xlabel('Time (minutes)', color='#6B6A66', fontsize=8)
    ax3.set_title('Breathing rate over time — 30s sliding window',
                  color='#E8E8E3', fontsize=9, pad=6)
    ax3.legend(fontsize=7, loc='upper right',
               facecolor='#2A2B30', labelcolor='#9B9A97',
               edgecolor='none')

    # ── narrative panel ───────────────────────────────────────────────
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Clinical Narrative Report',
                  color='#E8E8E3', fontsize=10, pad=8,
                  fontweight='normal')

    # wrap and display narrative text
    lines      = narrative.split('\n')
    y_pos      = 0.97
    line_h     = 0.026

    for line in lines:
        if y_pos < 0.01:
            break

        if line.startswith('===') or line.startswith('---'):
            y_pos -= line_h * 0.4
            continue

        # section headers
        is_header = (
            line.isupper() and len(line) > 3
            and not line.startswith('Patient')
            and not line.startswith('Recording')
            and not line.startswith('Signal')
            and not line.startswith('Method')
        )

        color    = '#E8E8E3'
        fontsize = 7.5
        weight   = 'normal'

        if is_header:
            color    = '#9FE1CB'
            fontsize = 8
            weight   = 'bold'
            y_pos   -= line_h * 0.3
        elif line.startswith('CLINICAL NOTE') or line.startswith('ELEVATED') or line.startswith('SLOW'):
            color = '#F5C4B3'
        elif line.startswith('Patient:') or line.startswith('Recording'):
            color = '#6B6A66'
            fontsize = 7

        if line.strip():
            # wrap long lines
            max_chars = 52
            words     = line.split()
            curr_line = ''
            for word in words:
                if len(curr_line) + len(word) + 1 <= max_chars:
                    curr_line += (' ' if curr_line else '') + word
                else:
                    ax4.text(0.03, y_pos, curr_line,
                             transform = ax4.transAxes,
                             color     = color,
                             fontsize  = fontsize,
                             fontweight= weight,
                             va        = 'top',
                             fontfamily= 'monospace')
                    y_pos    -= line_h
                    curr_line = word
            if curr_line:
                ax4.text(0.03, y_pos, curr_line,
                         transform = ax4.transAxes,
                         color     = color,
                         fontsize  = fontsize,
                         fontweight= weight,
                         va        = 'top',
                         fontfamily= 'monospace')

        y_pos -= line_h

    # ── title ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.96,
             f'Breathing Pattern Narrator — {patient_id.upper()} — '
             f'PPG-derived Respiratory Analysis',
             ha='center', color='#E8E8E3', fontsize=11,
             fontweight='normal')
    fig.text(0.5, 0.93,
             'Bandpass filter (0.1–0.5 Hz) · Peak detection · '
             'Rule-based narrative · No ML · For research use only',
             ha='center', color='#6B6A66', fontsize=8)

    plt.savefig(f'output/{patient_id}_full_report.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0F1117')
    print(f"Report saved to output/{patient_id}_full_report.png")
    plt.show()


if __name__ == '__main__':
    plot_full_report(PATIENT)
    