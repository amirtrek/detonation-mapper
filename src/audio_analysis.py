"""Audio extraction and explosion transient detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.signal import find_peaks

try:
    from moviepy import VideoFileClip
except ImportError:  # pragma: no cover - compatibility fallback
    from moviepy.editor import VideoFileClip

from models import AudioEventCandidate
from progress import progress, progress_kv


@dataclass
class AudioAnalysisOutput:
    """Audio analysis output with debug features."""

    sample_rate: int | None
    clip_duration_sec: float
    mono_samples: np.ndarray
    candidates: list[AudioEventCandidate]
    warnings: list[str]
    debug: dict[str, np.ndarray | list[float]]


def _normalize_waveform(samples: np.ndarray) -> np.ndarray:
    progress_kv("[audio] Normalizing waveform", sample_count=samples.size)
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(samples)))
    progress_kv("[audio] Waveform peak", peak=peak)
    if peak <= 1e-9:
        return samples.astype(np.float32, copy=True)
    return (samples / peak).astype(np.float32, copy=False)


def _frame_audio(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    progress_kv("[audio] Framing audio", sample_count=samples.size, frame_length=frame_length, hop_length=hop_length)
    if samples.size == 0:
        return np.empty((0, frame_length), dtype=np.float32)

    if samples.size <= frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        return padded.reshape(1, frame_length).astype(np.float32, copy=False)

    frame_count = 1 + math.ceil((samples.size - frame_length) / hop_length)
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(samples, (0, padded_length - samples.size))
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(frame_count, frame_length),
        strides=strides,
        writeable=False,
    )
    progress_kv("[audio] Created audio frames", frame_count=frame_count, padded_length=padded_length)
    return frames.astype(np.float32, copy=False)


def _compute_audio_features(
    mono: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    progress_kv(
        "[audio] Computing audio features",
        sample_count=mono.size,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    frames = _frame_audio(mono, frame_length=frame_length, hop_length=hop_length)
    if frames.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float32)).astype(np.float32, copy=False)

    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32, copy=False)
    if spectrum.shape[0] <= 1:
        onset = np.zeros(spectrum.shape[0], dtype=np.float32)
    else:
        positive_flux = np.maximum(np.diff(spectrum, axis=0), 0.0)
        onset = np.concatenate(([0.0], positive_flux.sum(axis=1, dtype=np.float32))).astype(np.float32, copy=False)

    times = (np.arange(frames.shape[0], dtype=np.float32) * hop_length) / float(sample_rate)
    progress_kv("[audio] Computed feature arrays", rms_count=rms.size, onset_count=onset.size, time_count=times.size)
    return rms, onset, times


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    progress_kv("[audio] Robust-normalizing array", size=values.size)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.std(values)) or 1.0
    progress_kv("[audio] Normalization stats", median=median, mad=mad, scale=scale)
    return (values - median) / scale


def _detect_onset_times(
    onset_score: np.ndarray,
    times: np.ndarray,
    min_distance_frames: int,
) -> list[float]:
    progress_kv("[audio] Detecting onset times", onset_size=onset_score.size, min_distance_frames=min_distance_frames)
    if onset_score.size == 0:
        return []

    prominence = max(0.25, float(np.std(onset_score)) * 0.25)
    onset_indices, _ = find_peaks(
        onset_score,
        distance=min_distance_frames,
        prominence=prominence,
    )
    progress_kv("[audio] Onset peaks located", onset_count=len(onset_indices), prominence=prominence)
    return times[onset_indices].tolist()


def _merge_event_times(
    candidate_times: list[float],
    combined_times: np.ndarray,
    combined_scores: np.ndarray,
    min_spacing_sec: float,
) -> list[tuple[float, float, int]]:
    progress_kv(
        "[audio] Merging event times",
        candidate_count=len(candidate_times),
        combined_count=combined_times.size,
        min_spacing_sec=min_spacing_sec,
    )
    merged: list[tuple[float, float, int]] = []
    for time_sec in sorted(candidate_times):
        nearest_index = int(np.argmin(np.abs(combined_times - time_sec)))
        score = float(combined_scores[nearest_index])
        progress_kv("[audio] Examining candidate time", time_sec=time_sec, nearest_index=nearest_index, score=score)
        if not merged or abs(time_sec - merged[-1][0]) >= min_spacing_sec:
            merged.append((time_sec, score, nearest_index))
            continue
        if score > merged[-1][1]:
            merged[-1] = (time_sec, score, nearest_index)
    return merged


def _prune_globally_weak_candidates(
    candidates: list[AudioEventCandidate],
    global_ratio_threshold: float = 0.10,
) -> list[AudioEventCandidate]:
    """Drop candidates whose score is far below the strongest boom.

    Explosion booms produce z-scores in the hundreds; background transients
    produce single-digit or low double-digit scores.  Neighbor-based pruning
    misses these because they are only compared to each other.
    """

    if not candidates:
        return candidates

    max_score = max(c.score for c in candidates)
    score_floor = max_score * global_ratio_threshold
    progress_kv(
        "[audio] Global weak-candidate pruning",
        candidate_count=len(candidates),
        max_score=max_score,
        global_ratio_threshold=global_ratio_threshold,
        score_floor=score_floor,
    )

    kept: list[AudioEventCandidate] = []
    pruned_count = 0
    for candidate in candidates:
        if candidate.score < score_floor:
            pruned_count += 1
            progress(
                f"[audio] Globally weak candidate at t={candidate.time_sec:.3f}s PRUNED "
                f"(score={candidate.score:.2f}, floor={score_floor:.2f}, "
                f"ratio={candidate.score / max_score:.5f}, "
                f"energy={candidate.energy_score:.2f}, onset={candidate.onset_score:.2f})."
            )
        else:
            kept.append(candidate)

    for candidate_id, candidate in enumerate(kept, start=1):
        candidate.candidate_id = candidate_id
    progress(f"[audio] Global pruning kept {len(kept)} candidate(s), removed {pruned_count}.")
    return kept


def _prune_weak_interior_candidates(
    candidates: list[AudioEventCandidate],
    relative_score_threshold: float = 0.10,
) -> list[AudioEventCandidate]:
    """Drop tiny interior transients that sit between much stronger candidates."""

    progress_kv(
        "[audio] Pruning weak interior candidates",
        candidate_count=len(candidates),
        relative_score_threshold=relative_score_threshold,
    )
    if len(candidates) <= 2:
        return candidates

    kept: list[AudioEventCandidate] = [candidates[0]]
    pruned_count = 0
    for index in range(1, len(candidates) - 1):
        candidate = candidates[index]
        previous_candidate = candidates[index - 1]
        next_candidate = candidates[index + 1]
        score_floor = min(previous_candidate.score, next_candidate.score) * relative_score_threshold
        if candidate.score < score_floor:
            pruned_count += 1
            progress(
                f"[audio] Pruned weak interior candidate at t={candidate.time_sec:.3f}s "
                f"(score={candidate.score:.2f}, floor={score_floor:.2f})."
            )
            continue
        kept.append(candidate)

    kept.append(candidates[-1])
    for candidate_id, candidate in enumerate(kept, start=1):
        candidate.candidate_id = candidate_id
    progress(f"[audio] Weak interior pruning kept {len(kept)} candidate(s), removed {pruned_count}.")
    return kept


def analyze_audio(
    video_path: str | Path,
    audio_threshold_z: float = 2.0,
    min_spacing_sec: float = 0.35,
    hop_length: int = 512,
) -> AudioAnalysisOutput:
    """Extract audio from an MP4 and detect boom-like transients."""

    warnings: list[str] = []
    frame_length = 2048
    progress_kv(
        "[audio] analyze_audio() called",
        video_path=str(video_path),
        audio_threshold_z=audio_threshold_z,
        min_spacing_sec=min_spacing_sec,
        hop_length=hop_length,
        frame_length=frame_length,
    )
    progress(f"[audio] Opening clip audio from {video_path}.")
    clip = VideoFileClip(str(video_path))
    try:
        duration = float(clip.duration or 0.0)
        progress(f"[audio] Clip duration: {duration:.2f}s.")
        if clip.audio is None:
            warnings.append("The MP4 has no audio track.")
            progress("[audio] No audio track found in the clip.")
            return AudioAnalysisOutput(
                sample_rate=None,
                clip_duration_sec=duration,
                mono_samples=np.array([], dtype=np.float32),
                candidates=[],
                warnings=warnings,
                debug={},
            )

        sample_rate = int(getattr(clip.audio, "fps", 44100) or 44100)
        progress(f"[audio] Extracting waveform at {sample_rate} Hz.")
        audio_array = clip.audio.to_soundarray(fps=sample_rate)
        progress(f"[audio] Extracted raw waveform with shape {audio_array.shape}.")
        mono = np.mean(audio_array, axis=1, dtype=np.float32) if audio_array.ndim > 1 else audio_array.astype(np.float32)
        if mono.size == 0:
            warnings.append("Audio extraction returned an empty waveform.")
            progress("[audio] Audio extraction returned an empty waveform.")
            return AudioAnalysisOutput(
                sample_rate=sample_rate,
                clip_duration_sec=duration,
                mono_samples=mono,
                candidates=[],
                warnings=warnings,
                debug={},
            )

        progress(f"[audio] Converted to mono with {mono.size} sample(s); normalizing waveform.")
        mono = _normalize_waveform(mono)
        progress("[audio] Computing RMS and onset features.")
        rms, onset, times = _compute_audio_features(
            mono,
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        progress(f"[audio] Computed {len(times)} audio frame(s) of features.")

        energy_score = _robust_normalize(rms)
        onset_score = _robust_normalize(onset)
        combined_score = 0.55 * energy_score + 0.45 * onset_score

        min_distance_frames = max(1, int(min_spacing_sec * sample_rate / hop_length))
        progress(
            f"[audio] Detecting peaks with min spacing {min_spacing_sec:.2f}s "
            f"({min_distance_frames} frame(s))."
        )
        peak_indices, properties = find_peaks(
            combined_score,
            height=np.median(combined_score) + audio_threshold_z,
            distance=min_distance_frames,
            prominence=0.5,
        )
        peak_times = times[peak_indices].tolist()
        progress_kv(
            "[audio] Peak detection found energy/onset peaks",
            peak_count=len(peak_times),
            threshold=float(np.median(combined_score) + audio_threshold_z),
        )
        onset_times = _detect_onset_times(onset_score, times, min_distance_frames)
        progress(f"[audio] Onset detection found {len(onset_times)} onset candidate(s).")

        merge_spacing_sec = max(min_spacing_sec, 4.5)
        merged = _merge_event_times(peak_times + onset_times, times, combined_score, merge_spacing_sec)
        progress(f"[audio] Merged detections into {len(merged)} audio candidate(s).")
        candidates: list[AudioEventCandidate] = []
        peak_heights = properties.get("peak_heights", np.array([], dtype=float))
        height_scale = float(np.max(peak_heights)) if peak_heights.size else max(float(np.max(combined_score)), 1.0)
        for candidate_id, (time_sec, score, array_index) in enumerate(merged, start=1):
            e_score = float(energy_score[array_index])
            o_score = float(onset_score[array_index])
            confidence = float(np.clip(score / max(height_scale, 1.0), 0.0, 1.0))
            candidates.append(
                AudioEventCandidate(
                    candidate_id=candidate_id,
                    time_sec=float(time_sec),
                    score=float(score),
                    energy_score=e_score,
                    onset_score=o_score,
                    confidence=confidence,
                )
            )
            progress(
                f"[audio] Candidate {candidate_id}: t={time_sec:.3f}s, score={score:.2f} "
                f"(energy={e_score:.2f}, onset={o_score:.2f})."
            )
        candidates = _prune_weak_interior_candidates(candidates)
        candidates = _prune_globally_weak_candidates(candidates)
        if not candidates:
            warnings.append("No strong audio transient candidates were detected.")
            progress("[audio] No strong audio transient candidates were detected.")
        else:
            progress(f"[audio] Audio analysis complete with {len(candidates)} candidate(s).")

        debug = {
            "times_sec": times,
            "rms": rms,
            "onset": onset,
            "energy_score": energy_score,
            "onset_score": onset_score,
            "combined_score": combined_score,
            "peak_indices": peak_indices,
        }
        return AudioAnalysisOutput(
            sample_rate=sample_rate,
            clip_duration_sec=duration,
            mono_samples=mono,
            candidates=candidates,
            warnings=warnings,
            debug=debug,
        )
    finally:
        progress("[audio] Closing moviepy clip handle.")
        clip.close()
