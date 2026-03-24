"""Map rendering for estimated explosion locations."""

from __future__ import annotations

from pathlib import Path

import folium

from geo import sector_points
from models import MatchedExplosionEvent
from progress import progress, progress_kv


def build_map(
    camera_lat: float,
    camera_lon: float,
    events: list[MatchedExplosionEvent],
    output_html: str | Path,
) -> None:
    """Render the camera and any estimated event geometries to an HTML map."""

    progress_kv(
        "[map] build_map() called",
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        event_count=len(events),
        output_html=str(output_html),
    )
    fmap = folium.Map(location=[camera_lat, camera_lon], zoom_start=12, control_scale=True)
    progress("[map] Created folium base map.")
    folium.Marker(
        [camera_lat, camera_lon],
        tooltip="Camera",
        popup="Camera position",
        icon=folium.Icon(color="blue", icon="camera"),
    ).add_to(fmap)
    progress("[map] Added camera marker.")

    exact_marker_count = 0
    sector_count = 0
    circle_count = 0

    for event in events:
        progress_kv(
            "[map] Rendering event",
            event_id=event.event_id,
            estimated_lat=event.estimated_lat,
            estimated_lon=event.estimated_lon,
            bearing_deg=event.bearing_deg,
            bearing_center_deg=event.bearing_center_deg,
            bearing_uncertainty_deg=event.bearing_uncertainty_deg,
        )
        popup_lines = [
            f"Event {event.event_id}",
            f"Visual: {event.visual_time_sec:.3f}s",
            f"Audio: {event.audio_time_sec:.3f}s",
            f"Delay: {event.delay_sec:.3f}s",
            f"Distance: {event.distance_m:.1f} m",
        ]
        if event.notes:
            popup_lines.extend(event.notes)

        if event.estimated_lat is not None and event.estimated_lon is not None and event.bearing_deg is not None:
            progress(f"[map] Adding exact marker and line for event {event.event_id}.")
            folium.Marker(
                [event.estimated_lat, event.estimated_lon],
                tooltip=f"Explosion {event.event_id}",
                popup="<br>".join(popup_lines),
                icon=folium.Icon(color="red", icon="warning-sign"),
            ).add_to(fmap)
            folium.PolyLine(
                [(camera_lat, camera_lon), (event.estimated_lat, event.estimated_lon)],
                color="#c0392b",
                weight=2,
                opacity=0.85,
            ).add_to(fmap)
            exact_marker_count += 1
            continue

        if event.bearing_center_deg is not None and event.bearing_uncertainty_deg:
            progress(f"[map] Adding uncertainty sector for event {event.event_id}.")
            spread_deg = event.bearing_uncertainty_deg * 2.0
            polygon = sector_points(
                lat=camera_lat,
                lon=camera_lon,
                center_bearing_deg=event.bearing_center_deg,
                spread_deg=spread_deg,
                distance_m=event.distance_m,
            )
            folium.Polygon(
                locations=polygon,
                color="#f39c12",
                fill=True,
                fill_opacity=0.20,
                popup="<br>".join(popup_lines + ["Uncertain bearing sector"]),
            ).add_to(fmap)
            sector_count += 1
            continue

        if event.bearing_center_deg is not None:
            progress(f"[map] Adding uncertainty circle for event {event.event_id}.")
            folium.Circle(
                location=[camera_lat, camera_lon],
                radius=event.distance_m,
                color="#f39c12",
                fill=False,
                popup="<br>".join(popup_lines + ["Distance known, bearing uncertainty unresolved"]),
            ).add_to(fmap)
            circle_count += 1

    progress(
        "[map] Geometry summary: "
        f"exact_markers={exact_marker_count}, "
        f"uncertainty_sectors={sector_count}, "
        f"uncertainty_circles={circle_count}."
    )

    output_path = Path(output_html)
    progress(f"[map] Ensuring output directory exists for {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"[map] Saving map HTML to {output_path}.")
    fmap.save(str(output_path))
    progress("[map] Map save complete.")
