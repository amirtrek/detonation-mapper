"""Video frame scanning and candidate visual explosion detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from models import MatchedExplosionEvent, VisualEventCandidate
from progress import progress, progress_kv


@dataclass
class VideoAnalysisOutput:
    """Video analysis output with debug series."""

    fps: float
    frame_count: int
    duration_sec: float
    frame_width: int
    frame_height: int
    candidates: list[VisualEventCandidate]
    warnings: list[str]
    debug: dict[str, np.ndarray]


def _merge_nearby_peaks(
    peak_indices: np.ndarray,
    combined_score: np.ndarray,
    merge_window_frames: int,
) -> np.ndarray:
    """Collapse nearby visual peaks into a single strongest representative peak."""

    if peak_indices.size <= 1:
        return peak_indices.astype(np.int32, copy=False)

    merged: list[int] = []
    current_cluster: list[int] = [int(peak_indices[0])]
    representative_peak = int(peak_indices[0])
    for raw_index in peak_indices[1:]:
        peak_index = int(raw_index)
        if peak_index - representative_peak <= merge_window_frames:
            current_cluster.append(peak_index)
            if float(combined_score[peak_index]) >= float(combined_score[representative_peak]):
                representative_peak = peak_index
            continue
        merged.append(representative_peak)
        current_cluster = [peak_index]
        representative_peak = peak_index

    merged.append(representative_peak)
    return np.array(merged, dtype=np.int32)


def _prune_weak_interior_candidates(
    candidates: list[VisualEventCandidate],
    relative_score_threshold: float = 0.10,
) -> list[VisualEventCandidate]:
    """Drop tiny interior peaks that are bracketed by much stronger neighbors."""

    progress_kv(
        "[video] Pruning weak interior candidates",
        candidate_count=len(candidates),
        relative_score_threshold=relative_score_threshold,
    )
    if len(candidates) <= 2:
        return candidates

    kept: list[VisualEventCandidate] = [candidates[0]]
    pruned_count = 0
    for index in range(1, len(candidates) - 1):
        candidate = candidates[index]
        previous_candidate = candidates[index - 1]
        next_candidate = candidates[index + 1]
        score_floor = min(previous_candidate.score, next_candidate.score) * relative_score_threshold
        if candidate.score < score_floor:
            pruned_count += 1
            progress(
                f"[video] Pruned weak interior candidate at t={candidate.time_sec:.3f}s "
                f"(score={candidate.score:.2f}, floor={score_floor:.2f})."
            )
            continue
        kept.append(candidate)

    kept.append(candidates[-1])
    for candidate_id, candidate in enumerate(kept, start=1):
        candidate.candidate_id = candidate_id
    progress(f"[video] Weak interior pruning kept {len(kept)} candidate(s), removed {pruned_count}.")
    return kept


def _prune_globally_weak_candidates(
    candidates: list[VisualEventCandidate],
    global_ratio_threshold: float = 0.01,
) -> list[VisualEventCandidate]:
    """Drop candidates whose score is far below the strongest peak.

    When real explosions produce z-scores in the thousands and false positives
    produce scores in the tens, neighbor-based pruning misses the false positives
    because they are only compared to each other.  This pass removes any candidate
    whose score is less than *global_ratio_threshold* times the maximum score.
    """

    if not candidates:
        return candidates

    max_score = max(c.score for c in candidates)
    score_floor = max_score * global_ratio_threshold
    progress_kv(
        "[video] Global weak-candidate pruning",
        candidate_count=len(candidates),
        max_score=max_score,
        global_ratio_threshold=global_ratio_threshold,
        score_floor=score_floor,
    )

    kept: list[VisualEventCandidate] = []
    pruned_count = 0
    for candidate in candidates:
        if candidate.score < score_floor:
            pruned_count += 1
            progress(
                f"[video] Globally weak candidate at t={candidate.time_sec:.3f}s PRUNED "
                f"(score={candidate.score:.2f}, floor={score_floor:.2f}, "
                f"ratio={candidate.score / max_score:.5f}, "
                f"brightness_score={candidate.brightness_score:.2f}, "
                f"diff_score={candidate.diff_score:.2f}, "
                f"motion_score={candidate.motion_score:.2f})."
            )
        else:
            kept.append(candidate)

    for candidate_id, candidate in enumerate(kept, start=1):
        candidate.candidate_id = candidate_id
    progress(f"[video] Global pruning kept {len(kept)} candidate(s), removed {pruned_count}.")
    return kept


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    progress_kv("[video] Normalizing array", size=values.size)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.std(values)) or 1.0
    progress_kv("[video] Normalization stats", median=median, mad=mad, scale=scale)
    return (values - median) / scale


def analyze_video(
    video_path: str | Path,
    visual_threshold_z: float = 2.5,
    min_spacing_sec: float = 0.35,
) -> VideoAnalysisOutput:
    """Scan video frames for abrupt flash/motion signatures."""

    progress(f"[video] Opening video file {video_path}.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    warnings: list[str] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = frame_count / fps if fps > 0 else 0.0
        if fps <= 0 or frame_count <= 0:
            raise ValueError("Video metadata is invalid or unreadable.")
        progress(
            f"[video] Opened clip: {frame_count} frame(s) at {fps:.2f} fps "
            f"({duration_sec:.2f}s, {frame_width}x{frame_height})."
        )

        brightness: list[float] = []
        diff_scores: list[float] = []
        motion_scores: list[float] = []
        centroid_x: list[float] = []

        previous_gray: np.ndarray | None = None
        frames_processed = 0
        next_report_percent = 5
        while True:
            success, frame = capture.read()
            if not success:
                progress("[video] Capture returned no more frames.")
                break

            frames_processed += 1
            percent = (frames_processed / frame_count) * 100.0
            while percent >= next_report_percent and next_report_percent <= 100:
                current_time_sec = (frames_processed - 1) / fps
                eta_sec = max(0.0, duration_sec - current_time_sec)
                progress(
                    f"[video] Reading progress {next_report_percent}% "
                    f"(frame {frames_processed}/{frame_count}, t={current_time_sec:.2f}s, eta={eta_sec:.2f}s)."
                )
                next_report_percent += 5

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(gray) / 255.0))
            if previous_gray is None:
                diff_scores.append(0.0)
                motion_scores.append(0.0)
                centroid_x.append(float(frame_width) / 2.0)
                previous_gray = gray
                continue

            abs_diff = cv2.absdiff(gray, previous_gray)
            diff_scores.append(float(np.mean(abs_diff) / 255.0))
            motion_mask = abs_diff > 25
            motion_scores.append(float(np.mean(motion_mask)))
            if np.any(motion_mask):
                moments = cv2.moments(motion_mask.astype(np.uint8))
                if moments["m00"] > 0:
                    centroid_x.append(float(moments["m10"] / moments["m00"]))
                else:
                    centroid_x.append(float(frame_width) / 2.0)
            else:
                centroid_x.append(float(frame_width) / 2.0)
            previous_gray = gray

        progress(f"[video] Finished reading {frames_processed} frame(s); computing visual feature scores.")
        brightness_array = np.array(brightness, dtype=np.float32)
        diff_array = np.array(diff_scores, dtype=np.float32)
        motion_array = np.array(motion_scores, dtype=np.float32)
        x_array = np.array(centroid_x, dtype=np.float32)

        window = max(3, int(round(fps * 0.2)))
        kernel = np.ones(window, dtype=np.float32) / window
        brightness_baseline = np.convolve(brightness_array, kernel, mode="same")
        brightness_spike = np.maximum(0.0, brightness_array - brightness_baseline)
        progress_kv("[video] Computed brightness baseline", window=window, sample_count=len(brightness_array))

        brightness_score = _robust_normalize(brightness_spike)
        diff_score = _robust_normalize(diff_array)
        motion_score = _robust_normalize(motion_array)
        combined_score = 0.40 * brightness_score + 0.35 * diff_score + 0.25 * motion_score

        min_peak_distance = max(1, int(min_spacing_sec * fps))
        peak_indices, properties = find_peaks(
            combined_score,
            height=np.median(combined_score) + visual_threshold_z,
            distance=min_peak_distance,
            prominence=0.75,
        )
        merge_window_frames = max(min_peak_distance, int(round(fps * 4.5)))
        merged_peak_indices = _merge_nearby_peaks(peak_indices, combined_score, merge_window_frames)
        progress_kv(
            "[video] Peak detection complete",
            peak_count=len(peak_indices),
            merged_peak_count=len(merged_peak_indices),
            min_peak_distance=min_peak_distance,
            threshold=float(np.median(combined_score) + visual_threshold_z),
        )

        candidates: list[VisualEventCandidate] = []
        height_scale = float(np.max(properties.get("peak_heights", np.array([1.0]))))
        for candidate_id, frame_index in enumerate(merged_peak_indices.tolist(), start=1):
            score = float(combined_score[frame_index])
            b_score = float(brightness_score[frame_index])
            d_score = float(diff_score[frame_index])
            m_score = float(motion_score[frame_index])
            confidence = float(np.clip(score / max(height_scale, 1.0), 0.0, 1.0))
            candidates.append(
                VisualEventCandidate(
                    candidate_id=candidate_id,
                    time_sec=frame_index / fps,
                    frame_index=frame_index,
                    score=score,
                    brightness_score=b_score,
                    diff_score=d_score,
                    motion_score=m_score,
                    confidence=confidence,
                    x_position=float(x_array[frame_index]) if frame_width > 0 else None,
                )
            )
            progress(
                f"[video] Candidate {candidate_id}: frame={frame_index}, "
                f"t={frame_index / fps:.3f}s, score={score:.2f} "
                f"(brightness={b_score:.2f}, diff={d_score:.2f}, motion={m_score:.2f})."
            )

        candidates = _prune_weak_interior_candidates(candidates)
        candidates = _prune_globally_weak_candidates(candidates)

        if not candidates:
            warnings.append("No strong visual explosion candidates were detected.")
            progress("[video] No strong visual explosion candidates were detected.")
        else:
            progress(f"[video] Visual analysis complete with {len(candidates)} candidate(s).")

        debug = {
            "times_sec": np.arange(len(combined_score), dtype=np.float32) / fps,
            "brightness": brightness_array,
            "brightness_spike": brightness_spike,
            "diff_score_raw": diff_array,
            "motion_score_raw": motion_array,
            "brightness_score": brightness_score,
            "diff_score": diff_score,
            "motion_score": motion_score,
            "combined_score": combined_score,
            "peak_indices": merged_peak_indices.astype(np.int32),
            "centroid_x": x_array,
        }
        return VideoAnalysisOutput(
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            frame_width=frame_width,
            frame_height=frame_height,
            candidates=candidates,
            warnings=warnings,
            debug=debug,
        )
    finally:
        progress("[video] Releasing OpenCV capture handle.")
        capture.release()


def read_frame_at_index(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read a specific frame by index."""

    progress_kv("[video] read_frame_at_index() called", video_path=str(video_path), frame_index=frame_index)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        progress(f"[video] Seeking to frame {frame_index}.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"Could not read frame {frame_index}")
        progress_kv("[video] Read frame successfully", frame_index=frame_index, frame_shape=tuple(frame.shape))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        progress("[video] Releasing OpenCV capture handle after random frame read.")
        capture.release()


def export_annotated_frames(
    video_path: str | Path,
    events: list[MatchedExplosionEvent],
    output_dir: str | Path,
    fps: float,
) -> None:
    """Save annotated still frames for matched events."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress(f"[video] Exporting {len(events)} annotated frame(s) to {output_path}.")
    for event in events:
        frame_index = max(0, int(round(event.visual_time_sec * fps)))
        progress(f"[video] Writing annotated frame for event {event.event_id} from frame {frame_index}.")
        frame = read_frame_at_index(video_path, frame_index)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if event.x_position is not None:
            x_position = int(round(event.x_position))
            cv2.line(bgr_frame, (x_position, 0), (x_position, bgr_frame.shape[0] - 1), (0, 255, 255), 2)
        label = f"Event {event.event_id} | {event.visual_time_sec:.2f}s -> {event.audio_time_sec:.2f}s"
        cv2.putText(
            bgr_frame,
            label,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path / f"event_{event.event_id:02d}.jpg"), bgr_frame)
    progress("[video] Annotated frame export complete.")
