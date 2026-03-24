from cli import _apply_geolocation
from models import MatchedExplosionEvent


def test_apply_geolocation_estimates_bearing_from_x_position() -> None:
    event = MatchedExplosionEvent(
        event_id=1,
        visual_candidate_id=1,
        audio_candidate_id=1,
        visual_time_sec=1.0,
        audio_time_sec=4.0,
        delay_sec=3.0,
        distance_m=1000.0,
        distance_km=1.0,
        confidence=0.9,
        x_position=500.0,
    )

    warnings = _apply_geolocation(
        events=[event],
        camera_lat=31.7,
        camera_lon=35.2,
        camera_azimuth=110.0,
        hfov=60.0,
        frame_width=1000,
    )

    assert warnings == []
    assert event.bearing_deg == 110.0
    assert event.estimated_lat is not None
    assert event.estimated_lon is not None
