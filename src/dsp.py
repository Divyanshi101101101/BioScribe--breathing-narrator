import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import welch
import matplotlib.pyplot as plt

# ── what this script does ──────────────────────────────────────────────
# extracts breathing signal hidden inside the PPG waveform
# using a bandpass filter at the respiratory frequency range
# validates extraction against the reference breathing signal
# computes breathing rate over time using a sliding window
# ──────────────────────────────────────────────────────────────────────

# breathing frequency range in Hz
# normal breathing: 12–20 breaths/min = 0.2–0.33 Hz
# we use a wider range to catch slow/fast breathing
RESP_LOW_HZ  = 0.1    # 6 breaths/min — very slow
RESP_HIGH_HZ = 0.5    # 30 breaths/min — fast breathing
FILTER_ORDER = 3      # butterworth filter order


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """
    applies a butterworth bandpass filter to isolate
    a specific frequency range from a signal.

    this is the core DSP operation of the whole project.
    it works by:
    1. designing a filter that passes only frequencies between lowcut and highcut
    2. applying it forwards and backwards (filtfilt) to avoid phase distortion
    """
    nyquist = fs / 2.0                          # nyquist frequency
    low     = lowcut  / nyquist                 # normalise to nyquist
    high    = highcut / nyquist

    # design butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # apply filter — filtfilt applies forward then backward
    # this gives zero phase distortion (important for peak detection)
    filtered = filtfilt(b, a, signal)
    return filtered


def extract_breathing_from_ppg(ppg, fs, reference=None):
    """
    extracts the breathing signal hidden inside the PPG.
    automatically corrects polarity if needed.
    """
    breathing = bandpass_filter(ppg, RESP_LOW_HZ, RESP_HIGH_HZ, fs)
    
    # if reference available, check polarity and flip if needed
    if reference is not None:
        corr = np.corrcoef(breathing[:len(reference)], 
                           reference[:len(breathing)])[0,1]
        if corr < 0:
            breathing = -breathing
    
    return breathing


def detect_breaths(breathing_signal, fs, min_breath_interval_sec=1.5):
    """
    detects individual breath peaks in the extracted breathing signal.

    each peak = one breath (the moment of maximum inspiration).
    min_breath_interval_sec prevents detecting noise as breaths —
    no human breathes faster than once every 1.5 seconds (40 breaths/min).
    """
    min_samples = int(min_breath_interval_sec * fs)

    peaks, properties = find_peaks(
        breathing_signal,
        distance  = min_samples,    # minimum gap between peaks
        prominence= 0.001           # minimum peak height above surroundings
    )
    return peaks


def compute_breathing_rate(breath_peaks, fs, window_sec=30, step_sec=10,
                            total_samples=None):
    """
    computes breathing rate (breaths per minute) over time
    using a sliding window.

    window_sec : how many seconds of data to use for each estimate
    step_sec   : how many seconds to move the window forward each step

    this gives us a breathing rate curve — not just one number,
    but how breathing rate changes over the whole recording.
    """
    window_samples = int(window_sec * fs)
    step_samples   = int(step_sec   * fs)

    times  = []   # centre time of each window (seconds)
    rates  = []   # breathing rate in that window (breaths/min)

    i = 0
    while i + window_samples <= total_samples:
        window_start = i
        window_end   = i + window_samples

        # count peaks that fall inside this window
        peaks_in_window = breath_peaks[
            (breath_peaks >= window_start) &
            (breath_peaks <  window_end)
        ]

        n_breaths    = len(peaks_in_window)
        rate_bpm     = n_breaths * (60.0 / window_sec)

        centre_time  = (window_start + window_samples // 2) / fs
        times.append(centre_time)
        rates.append(rate_bpm)

        i += step_samples

    return np.array(times), np.array(rates)


def compute_breathing_regularity(breath_peaks, fs):
    """
    computes how regular the breathing is.

    inter-breath interval (IBI) = time between consecutive breaths.
    coefficient of variation (CV) = std/mean of IBI.

    CV < 15% = very regular breathing
    CV 15–30% = moderately irregular
    CV > 30% = highly irregular breathing
    """
    if len(breath_peaks) < 3:
        return None, None, None

    ibi_samples = np.diff(breath_peaks)          # inter-breath intervals
    ibi_sec     = ibi_samples / fs               # convert to seconds

    mean_ibi = ibi_sec.mean()
    std_ibi  = ibi_sec.std()
    cv       = (std_ibi / mean_ibi) * 100        # coefficient of variation %

    return mean_ibi, std_ibi, cv


def validate_against_reference(extracted, reference, fs, patient_id):
    """
    compares our extracted breathing signal against the reference sensor.
    computes correlation — how well do we match ground truth?
    """
    # both signals need same length
    min_len     = min(len(extracted), len(reference))
    ext         = extracted[:min_len]
    ref         = reference[:min_len]

    # normalise both to 0-1 for fair comparison
    ext_norm    = (ext - ext.min()) / (ext.max() - ext.min())
    ref_norm    = (ref - ref.min()) / (ref.max() - ref.min())

    correlation = np.corrcoef(ext_norm, ref_norm)[0, 1]
    return correlation


def plot_extraction(ppg, breathing_extracted, breathing_reference,
                    breath_peaks, t, fs, patient_id):
    """plots the extraction result against ground truth"""

    # show first 60 seconds
    mask = t <= 60
    t_   = t[mask]

    # filter peaks to first 60 seconds
    peaks_60 = breath_peaks[breath_peaks < 60 * fs]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle(
        f'Breathing extraction from PPG — {patient_id.upper()} — first 60s',
        fontsize=13)

    # raw PPG
    axes[0].plot(t_, ppg[mask], color='#1D9E75', linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel('PPG (AU)')
    axes[0].set_title('Raw PPG — cardiac + respiratory components mixed')
    axes[0].set_xlim(0, 60)

    # extracted breathing vs reference
    ext_60 = breathing_extracted[mask]
    ref_60 = breathing_reference[mask]

    # normalise for visual comparison
    ext_norm = (ext_60 - ext_60.min()) / (ext_60.max() - ext_60.min())
    ref_norm = (ref_60 - ref_60.min()) / (ref_60.max() - ref_60.min())

    axes[1].plot(t_, ref_norm, color='#185FA5', linewidth=1.2,
                 label='Reference (ground truth)', alpha=0.8)
    axes[1].plot(t_, ext_norm, color='#E24B4A', linewidth=1.2,
                 label='Extracted from PPG', alpha=0.8, linestyle='--')
    axes[1].set_ylabel('Normalised amplitude')
    axes[1].set_title('Extracted breathing vs reference — validation')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(0, 60)

    # detected breath peaks
    axes[2].plot(t_, ext_norm, color='#E24B4A', linewidth=1, alpha=0.7)
    axes[2].scatter(peaks_60 / fs,
                    ext_norm[peaks_60],
                    color='#185FA5', zorder=5, s=40,
                    label=f'{len(peaks_60)} breaths detected')
    axes[2].set_ylabel('Breathing signal')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Individual breath detection')
    axes[2].legend(loc='upper right')
    axes[2].set_xlim(0, 60)

    plt.tight_layout()
    plt.savefig(f'output/{patient_id}_breathing_extraction.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to output/{patient_id}_breathing_extraction.png")


# ── test it ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from load_data import load_record, extract_signals

    PATIENT = 'bidmc01'

    print("Loading data...")
    record               = load_record(PATIENT)
    ppg, resp, hr, t, fs = extract_signals(record)

    print("Extracting breathing signal from PPG...")
    breathing = extract_breathing_from_ppg(ppg, fs, reference=resp)

    print("Detecting individual breaths...")
    breath_peaks = detect_breaths(breathing, fs)
    print(f"Total breaths detected : {len(breath_peaks)}")
    print(f"Recording duration     : {len(ppg)/fs:.0f} seconds")
    print(f"Mean breathing rate    : "
          f"{len(breath_peaks) / (len(ppg)/fs) * 60:.1f} breaths/min")

    print("\nComputing breathing regularity...")
    mean_ibi, std_ibi, cv = compute_breathing_regularity(breath_peaks, fs)
    print(f"Mean inter-breath interval : {mean_ibi:.2f} seconds")
    print(f"Std of IBI                 : {std_ibi:.2f} seconds")
    print(f"Coefficient of variation   : {cv:.1f}%")

    print("\nValidating against reference signal...")
    corr = validate_against_reference(breathing, resp, fs, PATIENT)
    print(f"Correlation with reference : {corr:.3f}")

    print("\nPlotting...")
    plot_extraction(ppg, breathing, resp, breath_peaks, t, fs, PATIENT)
