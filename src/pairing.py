"""Pair visual and audio explosion candidates."""

from __future__ import annotations

from dataclasses import replace

from geo import distance_from_delay
from models import AudioEventCandidate, MatchedExplosionEvent, VisualEventCandidate
from progress import progress, progress_kv


def _window_confidence(delay_sec: float, min_delay: float, max_delay: float) -> float:
    progress_kv(
        "[pairing] Computing window confidence",
        delay_sec=delay_sec,
        min_delay=min_delay,
        max_delay=max_delay,
    )
    if max_delay <= min_delay:
        return 0.0
    center = (min_delay + max_delay) / 2.0
    span = max((max_delay - min_delay) / 2.0, 1e-6)
    progress_kv("[pairing] Window confidence geometry", center=center, span=span)
    return max(0.0, 1.0 - abs(delay_sec - center) / span)


def _estimate_delay_cluster(
    visual_events: list[VisualEventCandidate],
    audio_events: list[AudioEventCandidate],
    min_delay: float,
    max_delay: float,
) -> tuple[float, float] | None:
    """Estimate the dominant visual-to-audio delay cluster."""

    weighted_delays: list[tuple[float, float]] = []
    audio_by_time = sorted(audio_events, key=lambda item: item.time_sec)
    for visual in sorted(visual_events, key=lambda item: item.time_sec):
        candidate_count = 0
        for audio in audio_by_time:
            delay_sec = audio.time_sec - visual.time_sec
            if delay_sec < min_delay:
                continue
            if delay_sec > max_delay:
                break
            weight = max(0.05, visual.confidence + audio.confidence)
            weighted_delays.append((delay_sec, weight))
            candidate_count += 1
            if candidate_count >= 3:
                break

    if not weighted_delays:
        return None

    weighted_delays.sort(key=lambda item: item[0])
    window_width = min(1.5, max(0.75, (max_delay - min_delay) * 0.08))
    best_start = 0
    best_end = 0
    best_weight = -1.0
    running_weight = 0.0
    end = 0

    for start in range(len(weighted_delays)):
        if end < start:
            end = start
            running_weight = 0.0
        while end < len(weighted_delays) and weighted_delays[end][0] - weighted_delays[start][0] <= window_width:
            running_weight += weighted_delays[end][1]
            end += 1
        if running_weight > best_weight:
            best_weight = running_weight
            best_start = start
            best_end = end
        running_weight -= weighted_delays[start][1]

    cluster = weighted_delays[best_start:best_end]
    total_weight = sum(weight for _, weight in cluster)
    if total_weight <= 0:
        return None

    target_delay = sum(delay * weight for delay, weight in cluster) / total_weight
    tolerance = max(1.0, window_width)
    return target_delay, tolerance


def build_matched_event(
    event_id: int,
    visual_event: VisualEventCandidate,
    audio_event: AudioEventCandidate,
    temp_c: float,
    min_delay: float,
    max_delay: float,
    extra_notes: list[str] | None = None,
) -> MatchedExplosionEvent:
    """Create a matched event from a visual/audio pair."""

    progress_kv(
        "[pairing] build_matched_event()",
        event_id=event_id,
        visual_candidate_id=visual_event.candidate_id,
        audio_candidate_id=audio_event.candidate_id,
    )
    delay_sec = audio_event.time_sec - visual_event.time_sec
    notes = list(extra_notes or [])
    if delay_sec < 0:
        notes.append("Audio precedes the visual event; pairing is physically suspect.")
    elif delay_sec < min_delay or delay_sec > max_delay:
        notes.append("Delay falls outside the configured auto-pairing window.")

    distance_m = max(0.0, distance_from_delay(max(delay_sec, 0.0), temp_c))
    confidence = max(
        0.0,
        min(
            1.0,
            0.45 * visual_event.confidence
            + 0.45 * audio_event.confidence
            + 0.10 * _window_confidence(delay_sec, min_delay, max_delay),
        ),
    )
    progress_kv(
        "[pairing] Built match metrics",
        delay_sec=delay_sec,
        distance_m=distance_m,
        confidence=confidence,
        note_count=len(notes),
    )

    return MatchedExplosionEvent(
        event_id=event_id,
        visual_candidate_id=visual_event.candidate_id,
        audio_candidate_id=audio_event.candidate_id,
        visual_time_sec=visual_event.time_sec,
        audio_time_sec=audio_event.time_sec,
        delay_sec=delay_sec,
        distance_m=distance_m,
        distance_km=distance_m / 1000.0,
        confidence=confidence,
        x_position=visual_event.x_position,
        notes=notes,
    )


def pair_events(
    visual_events: list[VisualEventCandidate],
    audio_events: list[AudioEventCandidate],
    temp_c: float,
    min_delay: float,
    max_delay: float,
) -> tuple[list[MatchedExplosionEvent], list[str]]:
    """Pair each visual event with the earliest plausible later audio event."""

    warnings: list[str] = []
    progress(
        f"[pairing] Starting pairing for {len(visual_events)} visual candidate(s) "
        f"and {len(audio_events)} audio candidate(s)."
    )
    if not visual_events:
        warnings.append("No visual explosion candidates were detected.")
        progress("[pairing] No visual explosion candidates were detected.")
        return [], warnings
    if not audio_events:
        warnings.append("No audio explosion candidates were detected.")
        progress("[pairing] No audio explosion candidates were detected.")
        return [], warnings

    audio_by_time = sorted(audio_events, key=lambda item: item.time_sec)
    preferred_delay = _estimate_delay_cluster(visual_events, audio_events, min_delay, max_delay)
    if preferred_delay is not None:
        progress(
            f"[pairing] Preferred delay cluster: target={preferred_delay[0]:.3f}s, "
            f"tolerance={preferred_delay[1]:.3f}s."
        )
    next_audio_start = 0
    matched: list[MatchedExplosionEvent] = []

    for visual in sorted(visual_events, key=lambda item: item.time_sec):
        progress(
            f"[pairing] Evaluating visual candidate {visual.candidate_id} "
            f"at t={visual.time_sec:.3f}s."
        )
        chosen_audio: AudioEventCandidate | None = None
        chosen_audio_index: int | None = None
        for audio_index in range(next_audio_start, len(audio_by_time)):
            audio = audio_by_time[audio_index]
            delay_sec = audio.time_sec - visual.time_sec
            if delay_sec < min_delay:
                continue
            if delay_sec > max_delay:
                break
            if preferred_delay is None:
                chosen_audio = audio
                chosen_audio_index = audio_index
                break

            target_delay, tolerance = preferred_delay
            if abs(delay_sec - target_delay) > tolerance:
                continue
            chosen_audio = audio
            chosen_audio_index = audio_index
            break

        if chosen_audio is None:
            warnings.append(
                f"Visual event at {visual.time_sec:.3f}s had no plausible audio match in "
                f"[{min_delay:.2f}, {max_delay:.2f}] seconds."
            )
            progress(
                f"[pairing] No plausible audio match found for visual candidate {visual.candidate_id}."
            )
            continue

        next_audio_start = (chosen_audio_index + 1) if chosen_audio_index is not None else next_audio_start
        progress(
            f"[pairing] Matched visual candidate {visual.candidate_id} "
            f"to audio candidate {chosen_audio.candidate_id} "
            f"(delay={chosen_audio.time_sec - visual.time_sec:.3f}s)."
        )
        matched.append(
            build_matched_event(
                event_id=len(matched) + 1,
                visual_event=visual,
                audio_event=chosen_audio,
                temp_c=temp_c,
                min_delay=min_delay,
                max_delay=max_delay,
            )
        )

    if not matched:
        warnings.append("No final visual/audio pairings were produced.")
        progress("[pairing] No final visual/audio pairings were produced.")
    else:
        progress(f"[pairing] Pairing finished with {len(matched)} matched event(s).")
    return matched, warnings


def apply_manual_x_positions(
    events: list[MatchedExplosionEvent],
    x_positions: list[float] | None,
) -> list[str]:
    """Apply user-provided x-positions to matched events in order."""

    if not x_positions:
        return []

    warnings: list[str] = []
    for index, event in enumerate(events):
        if index >= len(x_positions):
            warnings.append("Fewer x-positions were provided than matched events.")
            break
        event.x_position = x_positions[index]
        progress(f"[pairing] Applied manual x-position {x_positions[index]:.1f} to event {event.event_id}.")

    if len(x_positions) > len(events):
        warnings.append("More x-positions were provided than matched events; extras were ignored.")
    return warnings


def apply_exact_bearings(
    events: list[MatchedExplosionEvent],
    bearings: list[float] | None,
) -> list[str]:
    """Apply user-provided exact bearings to matched events in order."""

    if not bearings:
        return []

    warnings: list[str] = []
    for index, event in enumerate(events):
        if index >= len(bearings):
            warnings.append("Fewer exact bearings were provided than matched events.")
            break
        event.bearing_deg = bearings[index] % 360.0
        event.bearing_center_deg = event.bearing_deg
        event.bearing_uncertainty_deg = 0.0
        event.bearing_method = "exact"
        progress(f"[pairing] Applied exact bearing {event.bearing_deg:.2f} deg to event {event.event_id}.")

    if len(bearings) > len(events):
        warnings.append("More exact bearings were provided than matched events; extras were ignored.")
    return warnings


def replace_event(events: list[MatchedExplosionEvent], updated: MatchedExplosionEvent) -> list[MatchedExplosionEvent]:
    """Return a copy of an event list with one event replaced by matching event_id."""

    return [replace(item, **updated.to_dict()) if item.event_id == updated.event_id else item for item in events]
