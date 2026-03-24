"""Geospatial and distance utilities."""

from __future__ import annotations

from math import asin, atan2, cos, degrees, isfinite, radians, sin

from progress import progress, progress_kv

try:
    from pyproj import Geod
except ImportError:  # pragma: no cover - optional runtime fallback
    Geod = None

WGS84_GEOD = Geod(ellps="WGS84") if Geod is not None else None
EARTH_RADIUS_M = 6_378_137.0


def speed_of_sound(temp_c: float) -> float:
    """Return the approximate speed of sound in air in m/s."""

    result = 331.3 + 0.606 * temp_c
    progress_kv("[geo] speed_of_sound()", temp_c=temp_c, result=result)
    return result


def distance_from_delay(delay_sec: float, temp_c: float) -> float:
    """Convert audio delay to distance in meters."""

    progress_kv("[geo] distance_from_delay()", delay_sec=delay_sec, temp_c=temp_c)
    if delay_sec < 0:
        raise ValueError("delay_sec must be non-negative")
    result = delay_sec * speed_of_sound(temp_c)
    progress_kv("[geo] distance_from_delay() result", distance_m=result)
    return result


def normalize_bearing(bearing_deg: float) -> float:
    """Normalize a bearing into the 0-360 degree range."""

    result = bearing_deg % 360.0
    progress_kv("[geo] normalize_bearing()", bearing_deg=bearing_deg, result=result)
    return result


def bearing_from_x_position(
    x_position: float,
    frame_width: int,
    camera_azimuth_deg: float,
    hfov_deg: float,
) -> float:
    """Estimate the bearing of an event from its x-position in the frame."""

    progress_kv(
        "[geo] bearing_from_x_position()",
        x_position=x_position,
        frame_width=frame_width,
        camera_azimuth_deg=camera_azimuth_deg,
        hfov_deg=hfov_deg,
    )
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if hfov_deg <= 0:
        raise ValueError("hfov_deg must be positive")

    half_width = frame_width / 2.0
    offset_from_center_deg = ((x_position - half_width) / half_width) * (hfov_deg / 2.0)
    result = normalize_bearing(camera_azimuth_deg + offset_from_center_deg)
    progress_kv("[geo] bearing_from_x_position() result", half_width=half_width, offset_from_center_deg=offset_from_center_deg, result=result)
    return result


def destination_point(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """Project a destination point from a start point, bearing, and distance."""

    progress_kv(
        "[geo] destination_point()",
        lat=lat,
        lon=lon,
        bearing_deg=bearing_deg,
        distance_m=distance_m,
    )
    if not all(isfinite(value) for value in (lat, lon, bearing_deg, distance_m)):
        raise ValueError("All geodesic inputs must be finite values")
    if WGS84_GEOD is not None:
        dest_lon, dest_lat, _ = WGS84_GEOD.fwd(lon, lat, bearing_deg, distance_m)
        progress_kv("[geo] destination_point() used pyproj", dest_lat=dest_lat, dest_lon=dest_lon)
        return dest_lat, dest_lon

    angular_distance = distance_m / EARTH_RADIUS_M
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    bearing_rad = radians(bearing_deg)

    dest_lat_rad = asin(
        sin(lat_rad) * cos(angular_distance)
        + cos(lat_rad) * sin(angular_distance) * cos(bearing_rad)
    )
    dest_lon_rad = lon_rad + atan2(
        sin(bearing_rad) * sin(angular_distance) * cos(lat_rad),
        cos(angular_distance) - sin(lat_rad) * sin(dest_lat_rad),
    )
    dest_lat = degrees(dest_lat_rad)
    dest_lon = degrees(dest_lon_rad)
    progress_kv("[geo] destination_point() used spherical fallback", dest_lat=dest_lat, dest_lon=dest_lon)
    return dest_lat, dest_lon


def sector_points(
    lat: float,
    lon: float,
    center_bearing_deg: float,
    spread_deg: float,
    distance_m: float,
    point_count: int = 25,
) -> list[tuple[float, float]]:
    """Build a geodesic sector polygon for uncertainty rendering."""

    progress_kv(
        "[geo] sector_points()",
        lat=lat,
        lon=lon,
        center_bearing_deg=center_bearing_deg,
        spread_deg=spread_deg,
        distance_m=distance_m,
        point_count=point_count,
    )
    if spread_deg <= 0:
        raise ValueError("spread_deg must be positive")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")

    start_bearing = center_bearing_deg - spread_deg / 2.0
    step = spread_deg / (point_count - 1)
    points = [(lat, lon)]
    for index in range(point_count):
        bearing = start_bearing + step * index
        progress_kv("[geo] sector_points() computing vertex", index=index, bearing=bearing)
        points.append(destination_point(lat, lon, bearing, distance_m))
    points.append((lat, lon))
    progress(f"[geo] sector_points() built polygon with {len(points)} points.")
    return points
