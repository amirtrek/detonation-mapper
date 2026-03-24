"""Command-line interface for the Detonation Mapper analyzer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geo import (
    bearing_from_x_position,
    destination_point,
)
from models import AnalysisResult, MatchedExplosionEvent
from pairing import apply_exact_bearings, apply_manual_x_positions, pair_events
from progress import progress, progress_kv

if TYPE_CHECKING:
    from audio_analysis import AudioAnalysisOutput
    from video_analysis import VideoAnalysisOutput


def _parse_float_list(value: str | None) -> list[float] | None:
    progress_kv("[cli] Parsing float list", raw_value=value)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    parsed = [float(item.strip()) for item in stripped.split(",") if item.strip()]
    progress_kv("[cli] Parsed float list", parsed=parsed)
    return parsed


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    progress_kv("[cli] Preparing JSON write", path=str(output_path), keys=sorted(payload.keys()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    progress(f"[cli] JSON write finished for {output_path}.")


def _write_csv(path: str | Path, events: list[MatchedExplosionEvent]) -> None:
    output_path = Path(path)
    progress_kv("[cli] Preparing CSV write", path=str(output_path), event_count=len(events))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "visual_time_sec",
                "audio_time_sec",
                "delay_sec",
                "distance_m",
                "distance_km",
                "bearing_deg",
                "estimated_lat",
                "estimated_lon",
                "confidence",
            ],
        )
        writer.writeheader()
        for event in events:
            progress_kv(
                "[cli] Writing CSV row",
                event_id=event.event_id,
                visual_time_sec=event.visual_time_sec,
                audio_time_sec=event.audio_time_sec,
            )
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "visual_time_sec": event.visual_time_sec,
                    "audio_time_sec": event.audio_time_sec,
                    "delay_sec": event.delay_sec,
                    "distance_m": event.distance_m,
                    "distance_km": event.distance_km,
                    "bearing_deg": event.bearing_deg,
                    "estimated_lat": event.estimated_lat,
                    "estimated_lon": event.estimated_lon,
                    "confidence": event.confidence,
                }
            )
    progress(f"[cli] CSV write finished for {output_path}.")


def _print_console_summary(
    result: AnalysisResult,
    output_json: str | None,
    output_csv: str | None,
    output_map: str | None,
) -> None:
    print("Analysis summary")
    print(f"video: {result.video_path}")
    print(f"duration_sec: {result.clip_duration_sec:.2f}")
    print(f"visual_candidates: {len(result.candidate_visual_events)}")
    print(f"audio_candidates: {len(result.candidate_audio_events)}")
    print(f"final_pairings: {len(result.final_pairings)}")
    print(f"warnings: {len(result.warnings)}")
    for event in result.final_pairings:
        notes_text = "; ".join(event.notes) if event.notes else "-"
        bearing_text = f"{event.bearing_deg:.3f}" if event.bearing_deg is not None else "-"
        estimated_lat_text = f"{event.estimated_lat:.6f}" if event.estimated_lat is not None else "-"
        estimated_lon_text = f"{event.estimated_lon:.6f}" if event.estimated_lon is not None else "-"
        print(
            "event: "
            f"event_id={event.event_id}, "
            f"visual_time_sec={event.visual_time_sec:.3f}, "
            f"audio_time_sec={event.audio_time_sec:.3f}, "
            f"delay_sec={event.delay_sec:.3f}, "
            f"distance_m={event.distance_m:.1f}, "
            f"distance_km={event.distance_km:.3f}, "
            f"bearing_deg={bearing_text}, "
            f"estimated_lat={estimated_lat_text}, "
            f"estimated_lon={estimated_lon_text}, "
            f"confidence={event.confidence:.3f}, "
            f"notes={notes_text}"
        )
    if output_json:
        print(f"json created: {output_json}")
    if output_csv:
        print(f"csv created: {output_csv}")
    if output_map:
        print(f"map created: {output_map}")


def _save_debug_plots(
    video_output: VideoAnalysisOutput,
    audio_output: AudioAnalysisOutput,
    events: list[MatchedExplosionEvent],
    output_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_kv("[cli] Saving debug plots", output_dir=str(output_path), event_count=len(events))

    if video_output.debug:
        progress("[cli] Building visual debug plot.")
        video_times = np.asarray(video_output.debug["times_sec"])
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(video_times, video_output.debug["brightness_spike"], label="Brightness spike")
        axes[0].plot(video_times, video_output.debug["combined_score"], label="Combined score")
        axes[0].scatter(
            video_times[np.asarray(video_output.debug["peak_indices"], dtype=int)],
            np.asarray(video_output.debug["combined_score"])[np.asarray(video_output.debug["peak_indices"], dtype=int)],
            color="red",
            label="Detected peaks",
        )
        axes[0].legend()
        axes[0].set_ylabel("Visual score")

        axes[1].plot(video_times, video_output.debug["diff_score_raw"], label="Frame diff")
        axes[1].plot(video_times, video_output.debug["motion_score_raw"], label="Motion")
        for event in events:
            axes[1].axvline(event.visual_time_sec, color="green", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Raw visual features")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_path / "visual_debug.png", dpi=150)
        plt.close(figure)
        progress(f"[cli] Saved visual debug plot to {output_path / 'visual_debug.png'}.")

    if audio_output.debug:
        progress("[cli] Building audio debug plot.")
        audio_times = np.asarray(audio_output.debug["times_sec"])
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(audio_times, audio_output.debug["rms"], label="RMS")
        axes[0].plot(audio_times, audio_output.debug["onset"], label="Onset strength")
        axes[0].legend()
        axes[0].set_ylabel("Audio features")

        axes[1].plot(audio_times, audio_output.debug["combined_score"], label="Combined score")
        axes[1].scatter(
            audio_times[np.asarray(audio_output.debug["peak_indices"], dtype=int)],
            np.asarray(audio_output.debug["combined_score"])[np.asarray(audio_output.debug["peak_indices"], dtype=int)],
            color="red",
            label="Detected peaks",
        )
        for event in events:
            axes[1].axvline(event.audio_time_sec, color="green", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_path / "audio_debug.png", dpi=150)
        plt.close(figure)
        progress(f"[cli] Saved audio debug plot to {output_path / 'audio_debug.png'}.")


def _apply_geolocation(
    events: list[MatchedExplosionEvent],
    camera_lat: float | None,
    camera_lon: float | None,
    camera_azimuth: float | None,
    hfov: float | None,
    frame_width: int,
) -> list[str]:
    warnings: list[str] = []
    if camera_lat is None or camera_lon is None:
        progress("[geo] Camera latitude/longitude not provided; skipping geolocation.")
        return warnings

    exact_bearing_count = 0
    x_position_bearing_count = 0
    uncertain_sector_count = 0
    camera_centerline_only_count = 0
    insufficient_metadata_count = 0

    for event in events:
        if event.bearing_deg is not None:
            event.estimated_lat, event.estimated_lon = destination_point(
                lat=camera_lat,
                lon=camera_lon,
                bearing_deg=event.bearing_deg,
                distance_m=event.distance_m,
            )
            exact_bearing_count += 1
            continue

        if camera_azimuth is not None and event.x_position is not None and hfov is not None:
            event.bearing_deg = bearing_from_x_position(
                x_position=event.x_position,
                frame_width=frame_width,
                camera_azimuth_deg=camera_azimuth,
                hfov_deg=hfov,
            )
            event.bearing_center_deg = event.bearing_deg
            event.bearing_uncertainty_deg = 0.0
            event.bearing_method = "x_position"
            event.estimated_lat, event.estimated_lon = destination_point(
                lat=camera_lat,
                lon=camera_lon,
                bearing_deg=event.bearing_deg,
                distance_m=event.distance_m,
            )
            x_position_bearing_count += 1
            continue

        if camera_azimuth is not None:
            event.bearing_center_deg = camera_azimuth % 360.0
            if hfov is not None:
                event.bearing_uncertainty_deg = hfov / 2.0
                event.bearing_method = "camera_sector"
                event.notes.append(
                    "Bearing is uncertain across the camera field of view; map shows a sector, not a precise point."
                )
                uncertain_sector_count += 1
            else:
                event.bearing_method = "camera_centerline"
                event.notes.append(
                    "Camera azimuth is known but frame position/HFOV is missing, so no precise location was computed."
                )
                camera_centerline_only_count += 1
            continue

        event.notes.append("Insufficient bearing metadata for geolocation.")
        insufficient_metadata_count += 1

    progress(
        "[geo] Geolocation summary: "
        f"exact={exact_bearing_count}, "
        f"x_position={x_position_bearing_count}, "
        f"sector_uncertain={uncertain_sector_count}, "
        f"camera_centerline_only={camera_centerline_only_count}, "
        f"insufficient_metadata={insufficient_metadata_count}."
    )
    return warnings


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    progress("[cli] Building argument parser.")
    parser = argparse.ArgumentParser(
        description=(
            "Detect explosion flashes/booms in an MP4 clip, estimate distance from flash-to-boom delay, "
            "and optionally map the events."
        ),
        epilog=(
            "Minimum run: --video\n"
            "Map points: --video + --camera-lat + --camera-lon\n"
            "Directional mapping: add --camera-azimuth and --hfov"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required_group = parser.add_argument_group("Required input")
    required_group.add_argument(
        "--video",
        required=True,
        help="Required. Path to the MP4 clip to analyze.",
    )

    core_group = parser.add_argument_group("Core analysis")
    core_group.add_argument(
        "--temp-c",
        type=float,
        default=20.0,
        help="Optional. Ambient temperature in Celsius for the speed-of-sound calculation. Default: %(default)s.",
    )
    core_group.add_argument(
        "--min-delay",
        type=float,
        default=0.1,
        help="Optional. Minimum plausible delay in seconds between flash and boom. Default: %(default)s.",
    )
    core_group.add_argument(
        "--max-delay",
        type=float,
        default=20.0,
        help="Optional. Maximum plausible delay in seconds between flash and boom. Default: %(default)s.",
    )
    core_group.add_argument(
        "--visual-threshold-z",
        type=float,
        default=2.5,
        help="Optional. Visual detection threshold in robust z-score units. Higher means fewer visual candidates. Default: %(default)s.",
    )
    core_group.add_argument(
        "--audio-threshold-z",
        type=float,
        default=2.0,
        help="Optional. Audio detection threshold in robust z-score units. Higher means fewer audio candidates. Default: %(default)s.",
    )
    core_group.add_argument(
        "--min-spacing",
        type=float,
        default=0.35,
        help="Optional. Minimum spacing in seconds used during peak finding before later candidate condensation. Default: %(default)s.",
    )

    geo_group = parser.add_argument_group("Geolocation")
    geo_group.add_argument(
        "--camera-lat",
        type=float,
        default=None,
        help="Optional. Camera latitude in decimal degrees. Required only for map/geolocation output. Default: none.",
    )
    geo_group.add_argument(
        "--camera-lon",
        type=float,
        default=None,
        help="Optional. Camera longitude in decimal degrees. Required only for map/geolocation output. Default: none.",
    )
    geo_group.add_argument(
        "--camera-azimuth",
        type=float,
        default=None,
        help="Optional. Camera centerline compass bearing in degrees. Used with --hfov for directional mapping. Default: none.",
    )
    geo_group.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="Optional. Camera horizontal field of view in degrees. Used with --camera-azimuth and x-position to estimate bearing. Default: none.",
    )
    geo_group.add_argument(
        "--event-x-positions",
        default=None,
        help="Optional. Comma-separated per-event x pixel positions to override detected horizontal positions. Default: none.",
    )
    geo_group.add_argument(
        "--event-bearings",
        default=None,
        help="Optional. Comma-separated exact per-event bearings in degrees. Overrides x-position-derived bearings. Default: none.",
    )

    output_group = parser.add_argument_group("Outputs and review")
    output_group.add_argument(
        "--output-json",
        default=None,
        help="Optional. JSON results path. Default: none.",
    )
    output_group.add_argument(
        "--output-map",
        default=None,
        help="Optional. HTML map path. Requires --camera-lat and --camera-lon. Default: none.",
    )
    output_group.add_argument("--output-csv", default=None, help="Optional. CSV results path. Default: none.")
    output_group.add_argument("--debug-plots", action="store_true", help="Optional flag. Save visual/audio debug plots. Default: disabled.")
    output_group.add_argument(
        "--debug-dir",
        default="debug_plots",
        help="Optional. Directory for debug plot output. Default: %(default)s.",
    )
    output_group.add_argument(
        "--annotated-frames-dir",
        default=None,
        help="Optional. Directory for annotated still frames showing detected event positions. Default: none.",
    )
    output_group.add_argument(
        "--review-x-positions",
        action="store_true",
        help="Optional flag. Interactively click corrected x positions for matched events. Default: disabled.",
    )
    return parser


def run_analysis(args: argparse.Namespace) -> AnalysisResult:
    """Execute the full analysis pipeline."""

    from audio_analysis import analyze_audio
    from mapping import build_map
    from video_analysis import analyze_video, export_annotated_frames

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    progress_kv("[analysis] Parsed arguments", **vars(args))
    progress(f"[analysis] Starting analysis for {video_path}.")
    progress("[analysis] Scanning video frames for visual candidates...")
    video_output = analyze_video(
        video_path=video_path,
        visual_threshold_z=args.visual_threshold_z,
        min_spacing_sec=args.min_spacing,
    )
    progress(
        f"[analysis] Visual scan complete: found {len(video_output.candidates)} candidate(s) "
        f"across {video_output.duration_sec:.2f}s."
    )
    progress("[analysis] Scanning audio track for boom candidates...")
    audio_output = analyze_audio(
        video_path=video_path,
        audio_threshold_z=args.audio_threshold_z,
        min_spacing_sec=args.min_spacing,
    )
    progress(f"[analysis] Audio scan complete: found {len(audio_output.candidates)} candidate(s).")

    progress("[analysis] Pairing visual and audio candidates...")
    pairings, pairing_warnings = pair_events(
        visual_events=video_output.candidates,
        audio_events=audio_output.candidates,
        temp_c=args.temp_c,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )
    progress(f"[analysis] Pairing complete: produced {len(pairings)} matched event(s).")

    warnings = [*video_output.warnings, *audio_output.warnings, *pairing_warnings]

    if args.review_x_positions:
        from review import review_positions

        progress("[analysis] Entering interactive point review...")
        pairings, review_warnings = review_positions(
            video_path=video_path,
            visual_events=video_output.candidates,
            events=pairings,
            manual_x=args.review_x_positions,
        )
        warnings.extend(review_warnings)

    warnings.extend(apply_manual_x_positions(pairings, _parse_float_list(args.event_x_positions)))
    warnings.extend(apply_exact_bearings(pairings, _parse_float_list(args.event_bearings)))

    progress("[analysis] Applying geolocation metadata...")
    warnings.extend(
        _apply_geolocation(
            events=pairings,
            camera_lat=args.camera_lat,
            camera_lon=args.camera_lon,
            camera_azimuth=args.camera_azimuth,
            hfov=args.hfov,
            frame_width=video_output.frame_width,
        )
    )

    if args.debug_plots:
        progress(f"[analysis] Writing debug plots to {args.debug_dir}...")
        _save_debug_plots(video_output, audio_output, pairings, args.debug_dir)

    if args.annotated_frames_dir:
        progress(f"[analysis] Exporting annotated frames to {args.annotated_frames_dir}...")
        export_annotated_frames(
            video_path=video_path,
            events=pairings,
            output_dir=args.annotated_frames_dir,
            fps=video_output.fps,
        )

    if args.output_map:
        if args.camera_lat is None or args.camera_lon is None:
            raise ValueError("--output-map requires both --camera-lat and --camera-lon.")
        progress(f"[analysis] Building output map at {args.output_map}...")
        build_map(
            camera_lat=args.camera_lat,
            camera_lon=args.camera_lon,
            events=pairings,
            output_html=args.output_map,
        )

    result = AnalysisResult(
        video_path=str(video_path),
        clip_duration_sec=max(video_output.duration_sec, audio_output.clip_duration_sec),
        video_fps=video_output.fps,
        audio_sample_rate=audio_output.sample_rate,
        frame_width=video_output.frame_width,
        frame_height=video_output.frame_height,
        candidate_visual_events=video_output.candidates,
        candidate_audio_events=audio_output.candidates,
        final_pairings=pairings,
        warnings=warnings,
    )
    progress_kv(
        "[analysis] Built result object",
        final_pairings=len(result.final_pairings),
        warnings=len(result.warnings),
        clip_duration_sec=result.clip_duration_sec,
    )

    if args.output_json:
        progress(f"[analysis] Writing JSON results to {args.output_json}...")
        _write_json(args.output_json, result.to_dict())
    if args.output_csv:
        progress(f"[analysis] Writing CSV results to {args.output_csv}...")
        _write_csv(args.output_csv, pairings)
    progress("[analysis] Analysis complete.")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    progress_kv("[cli] main() invoked", argv=argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    progress_kv("[cli] Parsed namespace", **vars(args))
    if args.min_delay < 0 or args.max_delay <= 0 or args.max_delay <= args.min_delay:
        parser.error("Delay window must satisfy 0 <= min-delay < max-delay.")
    if args.camera_lat is None and args.camera_lon is not None:
        parser.error("--camera-lon requires --camera-lat.")
    if args.camera_lon is None and args.camera_lat is not None:
        parser.error("--camera-lat requires --camera-lon.")

    try:
        progress("[cli] Entering run_analysis().")
        result = run_analysis(args)
    except Exception as exc:
        progress_kv("[cli] Analysis raised exception", exc_type=type(exc).__name__, message=str(exc))
        parser.exit(status=1, message=f"Error: {exc}\n")

    progress("[cli] Writing console summary to stdout.")
    _print_console_summary(
        result,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_map=args.output_map,
    )
    progress("[cli] main() completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
