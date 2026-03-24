from geo import (
    bearing_from_x_position,
    destination_point,
    distance_from_delay,
    speed_of_sound,
)


def test_speed_of_sound_formula() -> None:
    assert round(speed_of_sound(20.0), 2) == 343.42


def test_distance_from_delay() -> None:
    assert round(distance_from_delay(3.0, 20.0), 2) == 1030.26


def test_bearing_from_x_position_center() -> None:
    assert bearing_from_x_position(960, 1920, 110.0, 60.0) == 110.0


def test_destination_point_moves_east() -> None:
    lat, lon = destination_point(31.7, 35.2, 90.0, 1000.0)
    assert abs(lat - 31.7) <= 0.01
    assert lon > 35.2
