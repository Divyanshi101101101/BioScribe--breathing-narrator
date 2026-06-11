import numpy as np

# ── what this script does ──────────────────────────────────────────────
# takes the raw breathing analysis results
# computes meaningful clinical features over time segments
# classifies each segment into a breathing pattern category
# these features feed directly into the narrative engine
# ──────────────────────────────────────────────────────────────────────

# clinical breathing rate thresholds (breaths/min)
NORMAL_LOW  = 12.0
NORMAL_HIGH = 20.0
TACHYPNEA   = 25.0   # fast breathing — clinically significant
BRADYPNEA   = 8.0    # very slow breathing — clinically significant

# regularity thresholds (coefficient of variation %)
REGULAR_CV     = 15.0
IRREGULAR_CV   = 30.0

# segment duration for narrative (seconds)
SEGMENT_SEC = 120    # analyse in 2-minute segments


def classify_breathing_rate(rate_bpm):
    """
    classifies a breathing rate into a clinical category.
    these are standard clinical definitions.
    """
    if rate_bpm < BRADYPNEA:
        return 'bradypnea',    'critically slow'
    elif rate_bpm < NORMAL_LOW:
        return 'slow',         'below normal range'
    elif rate_bpm <= NORMAL_HIGH:
        return 'normal',       'within normal range'
    elif rate_bpm <= TACHYPNEA:
        return 'elevated',     'mildly elevated'
    else:
        return 'tachypnea',    'significantly elevated'


def classify_regularity(cv_percent):
    """
    classifies breathing regularity from coefficient of variation.
    """
    if cv_percent < REGULAR_CV:
        return 'regular',     'consistent rhythm'
    elif cv_percent < IRREGULAR_CV:
        return 'variable',    'moderately variable rhythm'
    else:
        return 'irregular',   'highly irregular rhythm'


def detect_episodes(rate_times, rate_values, threshold_high=25,
                    threshold_low=10, min_duration_sec=20):
    """
    detects sustained episodes of abnormal breathing rate.

    an episode is a continuous period where breathing rate
    stays above or below a threshold for at least min_duration_sec.

    returns list of episode dicts with start, end, type, mean_rate.
    """
    episodes = []
    if len(rate_times) < 3:
        return episodes

    dt = rate_times[1] - rate_times[0]   # seconds per step

    in_episode    = False
    episode_type  = None
    episode_start = None
    episode_rates = []

    for i, (t, r) in enumerate(zip(rate_times, rate_values)):
        is_high = r > threshold_high
        is_low  = r < threshold_low

        if not in_episode:
            if is_high or is_low:
                in_episode    = True
                episode_type  = 'tachypnea' if is_high else 'bradypnea'
                episode_start = t
                episode_rates = [r]
        else:
            current_type = 'tachypnea' if is_high else ('bradypnea'
                           if is_low else None)

            if current_type == episode_type:
                episode_rates.append(r)
            else:
                # episode ended
                duration = t - episode_start
                if duration >= min_duration_sec:
                    episodes.append({
                        'start'    : episode_start,
                        'end'      : t,
                        'duration' : duration,
                        'type'     : episode_type,
                        'mean_rate': np.mean(episode_rates),
                        'max_rate' : np.max(episode_rates)
                    })
                in_episode = False

    return episodes


def segment_analysis(breath_peaks, fs, total_samples,
                     segment_sec=SEGMENT_SEC):
    """
    divides the recording into segments and analyses each one.

    returns a list of segment dicts with:
    - time range
    - breathing rate
    - regularity
    - rate classification
    - regularity classification
    """
    segments  = []
    seg_samp  = int(segment_sec * fs)
    n_segs    = total_samples // seg_samp

    for i in range(n_segs):
        start_samp = i * seg_samp
        end_samp   = start_samp + seg_samp
        start_sec  = start_samp / fs
        end_sec    = end_samp   / fs

        # peaks in this segment
        seg_peaks  = breath_peaks[
            (breath_peaks >= start_samp) &
            (breath_peaks <  end_samp)
        ]

        n_breaths  = len(seg_peaks)
        rate_bpm   = n_breaths * (60.0 / segment_sec)

        # regularity within segment
        if len(seg_peaks) >= 3:
            ibi         = np.diff(seg_peaks) / fs
            mean_ibi    = ibi.mean()
            std_ibi     = ibi.std()
            cv          = (std_ibi / mean_ibi) * 100
        else:
            cv          = 0.0

        rate_cat, rate_desc   = classify_breathing_rate(rate_bpm)
        reg_cat,  reg_desc    = classify_regularity(cv)

        segments.append({
            'segment'      : i + 1,
            'start_sec'    : start_sec,
            'end_sec'      : end_sec,
            'start_min'    : start_sec / 60,
            'end_min'      : end_sec   / 60,
            'n_breaths'    : n_breaths,
            'rate_bpm'     : rate_bpm,
            'cv_percent'   : cv,
            'rate_category': rate_cat,
            'rate_desc'    : rate_desc,
            'reg_category' : reg_cat,
            'reg_desc'     : reg_desc
        })

    return segments


def compute_overall_stats(breath_peaks, fs, total_samples):
    """computes overall statistics for the full recording"""
    duration_sec  = total_samples / fs
    duration_min  = duration_sec  / 60

    n_breaths     = len(breath_peaks)
    mean_rate     = n_breaths * (60.0 / duration_sec)

    if len(breath_peaks) >= 3:
        ibi       = np.diff(breath_peaks) / fs
        mean_ibi  = ibi.mean()
        std_ibi   = ibi.std()
        cv        = (std_ibi / mean_ibi) * 100
        min_ibi   = ibi.min()
        max_ibi   = ibi.max()
    else:
        mean_ibi = std_ibi = cv = min_ibi = max_ibi = 0.0

    rate_cat, rate_desc = classify_breathing_rate(mean_rate)
    reg_cat,  reg_desc  = classify_regularity(cv)

    return {
        'duration_min' : duration_min,
        'n_breaths'    : n_breaths,
        'mean_rate'    : mean_rate,
        'mean_ibi'     : mean_ibi,
        'std_ibi'      : std_ibi,
        'cv_percent'   : cv,
        'min_ibi'      : min_ibi,
        'max_ibi'      : max_ibi,
        'rate_category': rate_cat,
        'rate_desc'    : rate_desc,
        'reg_category' : reg_cat,
        'reg_desc'     : reg_desc
    }


# ── test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from load_data import load_record, extract_signals
    from dsp       import extract_breathing_from_ppg, detect_breaths

    PATIENT = 'bidmc01'
    record               = load_record(PATIENT)
    ppg, resp, hr, t, fs = extract_signals(record)
    breathing            = extract_breathing_from_ppg(ppg, fs, resp)
    breath_peaks         = detect_breaths(breathing, fs)

    print("\nOVERALL STATS:")
    stats = compute_overall_stats(breath_peaks, fs, len(ppg))
    for k, v in stats.items():
        print(f"  {k:15s}: {v}")

    print("\nSEGMENT ANALYSIS:")
    segments = segment_analysis(breath_peaks, fs, len(ppg))
    for seg in segments:
        print(f"  Segment {seg['segment']} "
              f"({seg['start_min']:.1f}–{seg['end_min']:.1f} min): "
              f"{seg['rate_bpm']:.1f} bpm — {seg['rate_desc']}, "
              f"CV {seg['cv_percent']:.1f}% — {seg['reg_desc']}")

    print("\nEPISODE DETECTION:")
    rate_times  = np.array([s['start_sec'] for s in segments])
    rate_values = np.array([s['rate_bpm']  for s in segments])
    episodes    = detect_episodes(rate_times, rate_values)
    if episodes:
        for ep in episodes:
            print(f"  {ep['type'].upper()} at {ep['start']:.0f}s–"
                  f"{ep['end']:.0f}s ({ep['duration']:.0f}s), "
                  f"mean {ep['mean_rate']:.1f} bpm")
    else:
        print("  No sustained abnormal episodes detected")
        