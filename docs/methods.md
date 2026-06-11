# Clinical Methods Document
**Project:** Breathing Pattern Narrator
**Date:** June 2026

---

## 1. Physiological Basis

Breathing modulates the PPG signal through three mechanisms:

1. **Respiratory sinus arrhythmia** — heart rate varies with breathing
   cycle, causing periodic changes in beat-to-beat PPG amplitude
2. **Baseline wander** — thoracic pressure changes during breathing
   cause slow drift in PPG baseline (0.1–0.5 Hz)
3. **Amplitude modulation** — lung volume changes affect venous return,
   modulating PPG pulse amplitude

This project exploits mechanism 2 — baseline wander — which produces
the strongest and most consistent respiratory signal in PPG.

---

## 2. Filter Design

**Type:** Zero-phase Butterworth bandpass filter
**Passband:** 0.1–0.5 Hz (6–30 breaths/min)
**Order:** 3
**Implementation:** scipy.signal.filtfilt (forward-backward application)

**Parameter justification:**
- Lower cutoff 0.1 Hz: captures very slow breathing (6 bpm) and
  excludes ultra-low frequency baseline drift
- Upper cutoff 0.5 Hz: captures fast breathing up to 30 bpm and
  excludes cardiac frequency (typically >1 Hz at 60+ bpm)
- Order 3: sufficient roll-off without excessive ringing artefacts
- filtfilt: eliminates phase distortion — critical for accurate
  peak timing

---

## 3. Breath Detection

**Method:** Prominence-filtered peak detection
**Minimum inter-peak distance:** 1.5 seconds (equivalent to 40 bpm max)
**Prominence threshold:** 0.001 (relative units)

Each detected peak corresponds to one inspiration maximum.
Inter-breath interval (IBI) is computed as the time between
consecutive peaks.

---

## 4. Feature Computation

**Breathing rate:** breaths detected in window × (60 / window_duration)
**Coefficient of variation:** (std(IBI) / mean(IBI)) × 100
**Episode detection:** sustained periods >20s above 25 bpm or below 10 bpm

---

## 5. Validation

PPG-derived breathing signal validated against reference chest
impedance sensor included in BIDMC dataset.
Pearson correlation coefficient used as primary validation metric.
Patient BIDMC01: r = 0.452

---

## 6. Limitations

- PPG motion artefacts degrade extraction quality
- Poor peripheral perfusion reduces signal quality
- Irregular cardiac rhythm (AF) confounds baseline wander
- Single-channel PPG less robust than dedicated respiratory sensor
- Results not validated across full 53-patient dataset