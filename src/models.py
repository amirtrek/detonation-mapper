"""Shared data models for the explosion analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VisualEventCandidate:
    """Candidate explosion timestamp derived from video features."""

    candidate_id: int
    time_sec: float
    frame_index: int
    score: float
    brightness_score: float
    diff_score: float
    motion_score: float
    confidence: float
    x_position: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioEventCandidate:
    """Candidate boom timestamp derived from audio features."""

    candidate_id: int
    time_sec: float
    score: float
    energy_score: float
    onset_score: float
    confidence: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchedExplosionEvent:
    """Resolved visual/audio pairing with distance and optional geolocation."""

    event_id: int
    visual_candidate_id: int
    audio_candidate_id: int
    visual_time_sec: float
    audio_time_sec: float
    delay_sec: float
    distance_m: float
    distance_km: float
    confidence: float
    x_position: float | None = None
    bearing_deg: float | None = None
    bearing_center_deg: float | None = None
    bearing_uncertainty_deg: float | None = None
    bearing_method: str | None = None
    estimated_lat: float | None = None
    estimated_lon: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Top-level analysis payload returned by the CLI."""

    video_path: str
    clip_duration_sec: float
    video_fps: float
    audio_sample_rate: int | None
    frame_width: int
    frame_height: int
    candidate_visual_events: list[VisualEventCandidate]
    candidate_audio_events: list[AudioEventCandidate]
    final_pairings: list[MatchedExplosionEvent]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "clip_duration_sec": self.clip_duration_sec,
            "video_fps": self.video_fps,
            "audio_sample_rate": self.audio_sample_rate,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "candidate_visual_event_count": len(self.candidate_visual_events),
            "candidate_audio_event_count": len(self.candidate_audio_events),
            "final_pairings": [item.to_dict() for item in self.final_pairings],
            "warning_count": len(self.warnings),
        }
