import numpy as np

from audio_analysis import (
    _compute_audio_features,
    _frame_audio,
    _merge_event_times,
    _normalize_waveform,
    _prune_weak_interior_candidates,
)
from models import AudioEventCandidate


def test_normalize_waveform_scales_to_unit_peak() -> None:
    samples = np.array([0.0, 2.0, -4.0, 1.0], dtype=np.float32)
    normalized = _normalize_waveform(samples)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_frame_audio_pads_short_waveforms() -> None:
    frames = _frame_audio(np.array([1.0, -1.0], dtype=np.float32), frame_length=4, hop_length=2)
    assert frames.shape == (1, 4)
    assert np.allclose(frames[0], np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32))


def test_compute_audio_features_returns_aligned_series() -> None:
    samples = np.zeros(4096, dtype=np.float32)
    samples[1536:1664] = 1.0

    rms, onset, times = _compute_audio_features(
        samples,
        sample_rate=2048,
        frame_length=512,
        hop_length=128,
    )

    assert len(rms) == len(onset) == len(times)
    assert len(rms) > 0
    assert float(np.max(rms)) > 0.0
    assert float(np.max(onset)) > 0.0


def test_merge_event_times_keeps_strongest_candidate_in_cluster() -> None:
    combined_times = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    combined_scores = np.array([1.0, 5.0, 3.0, 4.0], dtype=np.float32)

    merged = _merge_event_times([0.0, 0.5, 1.0, 2.0], combined_times, combined_scores, min_spacing_sec=1.5)

    assert merged == [(0.5, 5.0, 1), (2.0, 4.0, 3)]


def test_merge_event_times_merges_chain_of_close_candidates() -> None:
    combined_times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    combined_scores = np.array([2.0, 6.0, 4.0], dtype=np.float32)

    merged = _merge_event_times([0.0, 1.0, 2.0], combined_times, combined_scores, min_spacing_sec=2.5)

    assert merged == [(1.0, 6.0, 1)]


def test_prune_weak_interior_audio_candidates_removes_tiny_middle_peak() -> None:
    candidates = [
        AudioEventCandidate(1, 1.0, 90.0, 0.0, 0.0, 0.9),
        AudioEventCandidate(2, 2.0, 2.0, 0.0, 0.0, 0.1),
        AudioEventCandidate(3, 3.0, 110.0, 0.0, 0.0, 1.0),
        AudioEventCandidate(4, 4.0, 4.0, 0.0, 0.0, 0.2),
    ]

    pruned = _prune_weak_interior_candidates(candidates)

    assert [candidate.candidate_id for candidate in pruned] == [1, 2, 3]
    assert [candidate.time_sec for candidate in pruned] == [1.0, 3.0, 4.0]
