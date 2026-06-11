import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── what this script does ──────────────────────────────────────────────
# loads one BIDMC patient record
# prints exactly what signals are available and their properties
# plots the raw PPG and reference breathing signal side by side
# this is your first look at the real data before any processing
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path('data/raw')
PATIENT  = 'bidmc01'


def load_record(patient_id):
    """loads a BIDMC record and returns signals + metadata"""
    path   = DATA_DIR / patient_id
    record = wfdb.rdrecord(str(path))

    print("=" * 55)
    print(f"PATIENT: {patient_id.upper()}")
    print("=" * 55)
    print(f"Sampling frequency : {record.fs} Hz")
    print(f"Total samples      : {record.sig_len}")
    print(f"Duration           : {record.sig_len / record.fs:.1f} seconds "
          f"({record.sig_len / record.fs / 60:.1f} minutes)")
    print(f"Number of signals  : {record.n_sig}")
    print()
    print("Available signals:")
    for i, name in enumerate(record.sig_name):
        units = record.units[i]
        col   = record.p_signal[:, i]
        print(f"  [{i}] {name:10s} | units: {units:6s} | "
              f"mean: {np.nanmean(col):8.3f} | "
              f"std: {np.nanstd(col):8.3f}")

    return record


def extract_signals(record):
    """extracts PPG and reference breathing signal by name"""
    # clean signal names — strip quotes, commas, whitespace
    names = [n.strip().strip("'").strip(",").strip() 
             for n in record.sig_name]
    print(f"Cleaned signal names: {names}")

    ppg_idx  = names.index('PLETH')
    resp_idx = names.index('RESP')

    ppg  = record.p_signal[:, ppg_idx]
    resp = record.p_signal[:, resp_idx]
    hr   = np.zeros(len(ppg))

    fs = record.fs
    t  = np.arange(len(ppg)) / fs

    return ppg, resp, hr, t, fs

def plot_raw_signals(ppg, resp, hr, t, patient_id):
    """plots raw PPG, reference breathing, and heart rate"""

    # show first 60 seconds for clarity
    mask = t <= 60
    t_   = t[mask]
    ppg_ = ppg[mask]
    resp_= resp[mask]
    hr_  = hr[mask]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle(f'Raw signals — {patient_id.upper()} — first 60 seconds',
                 fontsize=13, fontweight='normal')

    # PPG
    axes[0].plot(t_, ppg_, color='#1D9E75', linewidth=0.8)
    axes[0].set_ylabel('PPG (AU)')
    axes[0].set_title('Photoplethysmography (PPG) — raw waveform')
    axes[0].set_xlim(0, 60)

    # Reference breathing
    axes[1].plot(t_, resp_, color='#185FA5', linewidth=1)
    axes[1].set_ylabel('Resp (AU)')
    axes[1].set_title('Reference breathing signal — ground truth')
    axes[1].set_xlim(0, 60)

    # Heart rate
    axes[2].plot(t_, hr_, color='#E24B4A', linewidth=1)
    axes[2].set_ylabel('HR (bpm)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Heart rate')
    axes[2].set_xlim(0, 60)

    plt.tight_layout()
    plt.savefig(f'output/{patient_id}_raw_signals.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to output/{patient_id}_raw_signals.png")


# ── run ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    record               = load_record(PATIENT)
    ppg, resp, hr, t, fs = extract_signals(record)
    plot_raw_signals(ppg, resp, hr, t, PATIENT)
    

    