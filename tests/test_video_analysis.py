import numpy as np

from models import VisualEventCandidate
from video_analysis import _merge_nearby_peaks, _prune_weak_interior_candidates


def test_merge_nearby_peaks_keeps_strongest_peak_in_cluster() -> None:
    peak_indices = np.array([10, 16, 40, 66], dtype=np.int32)
    combined_score = np.zeros(80, dtype=np.float32)
    combined_score[10] = 3.0
    combined_score[16] = 5.0
    combined_score[40] = 4.0
    combined_score[66] = 6.0

    merged = _merge_nearby_peaks(peak_indices, combined_score, merge_window_frames=30)

    assert merged.tolist() == [16, 66]


def test_merge_nearby_peaks_uses_strongest_peak_as_cluster_anchor() -> None:
    peak_indices = np.array([100, 130, 165], dtype=np.int32)
    combined_score = np.zeros(220, dtype=np.float32)
    combined_score[100] = 4.0
    combined_score[130] = 7.0
    combined_score[165] = 5.0

    merged = _merge_nearby_peaks(peak_indices, combined_score, merge_window_frames=40)

    assert merged.tolist() == [130]


def test_prune_weak_interior_candidates_removes_tiny_middle_peak() -> None:
    candidates = [
        VisualEventCandidate(1, 1.0, 10, 100.0, 0.0, 0.0, 0.0, 1.0),
        VisualEventCandidate(2, 2.0, 20, 3.0, 0.0, 0.0, 0.0, 0.1),
        VisualEventCandidate(3, 3.0, 30, 120.0, 0.0, 0.0, 0.0, 1.0),
        VisualEventCandidate(4, 4.0, 40, 8.0, 0.0, 0.0, 0.0, 0.2),
    ]

    pruned = _prune_weak_interior_candidates(candidates)

    assert [candidate.candidate_id for candidate in pruned] == [1, 2, 3]
    assert [candidate.time_sec for candidate in pruned] == [1.0, 3.0, 4.0]
