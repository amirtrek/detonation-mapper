from models import AudioEventCandidate, VisualEventCandidate
from pairing import pair_events


def test_no_visual_events_returns_warning() -> None:
    pairings, warnings = pair_events([], [AudioEventCandidate(1, 2.0, 3.0, 2.0, 2.0, 0.7)], 20.0, 0.1, 3.0)
    assert pairings == []
    assert "No visual explosion candidates were detected." in warnings


def test_overlapping_events_still_pair_in_order() -> None:
    visuals = [
        VisualEventCandidate(1, 1.0, 30, 4.0, 4.0, 3.0, 3.0, 0.8),
        VisualEventCandidate(2, 1.4, 42, 4.0, 4.0, 3.0, 3.0, 0.8),
    ]
    audios = [
        AudioEventCandidate(1, 1.7, 4.0, 4.0, 3.0, 0.8),
        AudioEventCandidate(2, 2.1, 4.0, 4.0, 3.0, 0.8),
    ]
    pairings, warnings = pair_events(visuals, audios, 20.0, 0.1, 2.0)
    assert not warnings
    assert len(pairings) == 2
    assert pairings[0].audio_candidate_id == 1
    assert pairings[1].audio_candidate_id == 2
