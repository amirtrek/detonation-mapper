from models import AudioEventCandidate, VisualEventCandidate
from pairing import build_matched_event, pair_events


def make_visual(candidate_id: int, time_sec: float) -> VisualEventCandidate:
    return VisualEventCandidate(
        candidate_id=candidate_id,
        time_sec=time_sec,
        frame_index=int(time_sec * 30),
        score=5.0,
        brightness_score=5.0,
        diff_score=4.0,
        motion_score=3.0,
        confidence=0.9,
    )


def make_audio(candidate_id: int, time_sec: float) -> AudioEventCandidate:
    return AudioEventCandidate(
        candidate_id=candidate_id,
        time_sec=time_sec,
        score=4.0,
        energy_score=4.0,
        onset_score=4.0,
        confidence=0.8,
    )


def test_pair_events_uses_earliest_plausible_unmatched_audio() -> None:
    visual_events = [make_visual(1, 1.0), make_visual(2, 4.0)]
    audio_events = [make_audio(1, 2.0), make_audio(2, 2.8), make_audio(3, 5.0)]
    pairings, warnings = pair_events(visual_events, audio_events, temp_c=20.0, min_delay=0.2, max_delay=3.0)
    assert not warnings
    assert [event.audio_candidate_id for event in pairings] == [1, 3]


def test_pair_events_handles_missing_audio() -> None:
    pairings, warnings = pair_events([make_visual(1, 1.0)], [], temp_c=20.0, min_delay=0.2, max_delay=3.0)
    assert pairings == []
    assert "No audio explosion candidates were detected." in warnings


def test_pair_events_preserves_sound_sequence() -> None:
    visual_events = [make_visual(1, 1.0), make_visual(2, 2.0)]
    audio_events = [make_audio(1, 2.2), make_audio(2, 3.0), make_audio(3, 3.2)]

    pairings, warnings = pair_events(visual_events, audio_events, temp_c=20.0, min_delay=0.1, max_delay=3.0)

    assert not warnings
    assert [event.audio_candidate_id for event in pairings] == [1, 2]


def test_build_matched_event_flags_negative_delay() -> None:
    event = build_matched_event(
        event_id=1,
        visual_event=make_visual(1, 5.0),
        audio_event=make_audio(1, 4.5),
        temp_c=20.0,
        min_delay=0.2,
        max_delay=3.0,
    )
    assert event.delay_sec < 0
    assert event.distance_m == 0.0
    assert any("physically suspect" in note for note in event.notes)


def test_pair_events_skips_far_outlier_delays_outside_dominant_cluster() -> None:
    visual_events = [make_visual(1, 1.0), make_visual(2, 2.0), make_visual(3, 3.0)]
    audio_events = [make_audio(1, 6.3), make_audio(2, 7.5), make_audio(3, 11.8)]

    pairings, warnings = pair_events(visual_events, audio_events, temp_c=20.0, min_delay=0.1, max_delay=20.0)

    assert [event.audio_candidate_id for event in pairings] == [1, 2]
    assert any("3.000s had no plausible audio match" in warning for warning in warnings)
