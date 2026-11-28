import numpy as np
from scipy.signal import butter, sosfilt

from multi_video_studio.audio_alignment import estimate_audio_clock, estimate_onset_alignment


def _events(sr=8000, seconds=12):
    rng = np.random.default_rng(120)
    t = np.arange(sr*seconds)/sr
    envelope = np.zeros(len(t))
    for time, width in zip(np.cumsum(rng.uniform(.25, .8, 35)), rng.uniform(.025, .09, 35)):
        envelope += np.exp(-.5*((t-time)/width)**2)*rng.uniform(.3, 1.)
    return t, envelope


def test_spectral_onsets_align_different_microphone_frequency_responses():
    sr = 8000
    t, envelope = _events(sr)
    # Same events, with almost disjoint carrier frequencies after microphone EQ.
    ref = envelope*(np.sin(2*np.pi*350*t) + .002*np.sin(2*np.pi*1900*t))
    target = envelope*(.002*np.sin(2*np.pi*350*t) + np.sin(2*np.pi*1900*t))
    target = np.r_[np.zeros(1096), target[:-1096]]  # 137 ms known delay.
    target = sosfilt(butter(2, 2500, fs=sr, output="sos"), target)
    report = estimate_onset_alignment(ref, target, sr, 1.)
    assert report["usable"], report
    assert abs(report["lag_s"]-.137) < .008, report
    details = {}
    offset, score, slope, _ = estimate_audio_clock(ref, target, sr, 1., details)
    assert abs(offset+.137) < .008 and score >= 10 and slope == 1, details


def test_repeated_beats_are_ambiguous_and_do_not_approve_a_clock():
    sr = 8000
    t = np.arange(sr*12)/sr
    envelope = np.maximum(0, np.sin(2*np.pi*2*t))**8
    audio = envelope*np.sin(2*np.pi*600*t)
    target = np.roll(audio, sr//4)
    report = estimate_onset_alignment(audio, target, sr, 2.)
    assert report["ambiguous"] and not report["usable"], report
    _, score, _, _ = estimate_audio_clock(audio, target, sr, 2.)
    assert score == 0


def test_silence_and_unrelated_events_cannot_create_an_onset_match():
    sr = 8000
    a = np.zeros(sr*8)
    assert not estimate_onset_alignment(a, a, sr, 1.)["usable"]
    t, env = _events(sr, 8)
    rng = np.random.default_rng(991)
    unrelated = rng.normal(size=len(t))
    assert not estimate_onset_alignment(env*np.sin(2*np.pi*400*t), unrelated, sr, 1.)["usable"]


def test_waveform_precision_is_retained_when_no_spectral_fallback_is_needed():
    sr = 16000
    rng = np.random.default_rng(31)
    ref = rng.normal(size=sr*2)
    target = np.r_[np.zeros(296), ref[:-296]]
    offset, score, _, _ = estimate_audio_clock(ref, target, sr, 1.)
    assert abs(offset+.0185) < .0001 and score >= 10


def test_one_clap_with_a_large_start_offset_is_not_mistaken_for_repeated_music():
    sr = 8000
    rng = np.random.default_rng(84)
    burst = rng.normal(size=800)*np.hanning(800)
    ref, target = np.zeros(sr*11), np.zeros(sr*4)
    ref[68000:68800], target[10400:11200] = burst, burst
    details = {}
    offset, score, _, _ = estimate_audio_clock(ref,target,sr,10.,details)
    assert abs(offset-7.2) < .001 and score >= 10, details
    assert not details["onset_check"]["ambiguous"]
