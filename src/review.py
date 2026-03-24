"""Interactive review helpers for manual pairing and point capture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from models import AudioEventCandidate, MatchedExplosionEvent, VisualEventCandidate
from pairing import build_matched_event
from progress import progress, progress_kv
from video_analysis import read_frame_at_index


def print_candidate_summary(
    visual_events: list[VisualEventCandidate],
    audio_events: list[AudioEventCandidate],
) -> None:
    """Print a compact summary of auto-detected candidates."""

    progress_kv(
        "[review] Printing candidate summary",
        visual_count=len(visual_events),
        audio_count=len(audio_events),
    )
    print("\nDetected candidates summary:")
    print(f"  visual candidates: {len(visual_events)}")
    print(f"  audio candidates: {len(audio_events)}")


def _select_position(video_path: str | Path, frame_index: int, event_id: int) -> tuple[float, float] | None:
    progress(
        f"[review] Opening frame picker for event {event_id} at frame {frame_index}. "
        "Click the explosion position, then close the window to continue."
    )
    frame = read_frame_at_index(video_path, frame_index)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(frame)
    axis.set_title(f"Click explosion position for event {event_id}, then close the window")
    axis.set_axis_off()
    clicked = plt.ginput(1, timeout=0)
    plt.close(figure)
    if not clicked:
        progress(f"[review] No position selected for event {event_id}; continuing without a manual override.")
        return None
    progress(
        f"[review] Selected point ({clicked[0][0]:.1f}, {clicked[0][1]:.1f}) for event {event_id}."
    )
    return float(clicked[0][0]), float(clicked[0][1])


def review_pairings(
    video_path: str | Path,
    visual_events: list[VisualEventCandidate],
    audio_events: list[AudioEventCandidate],
    current_pairings: list[MatchedExplosionEvent],
    temp_c: float,
    min_delay: float,
    max_delay: float,
    manual_x: bool = False,
) -> tuple[list[MatchedExplosionEvent], list[str]]:
    """Let the user confirm or override pairings and x-positions."""

    progress_kv(
        "[review] review_pairings() called",
        video_path=str(video_path),
        visual_count=len(visual_events),
        audio_count=len(audio_events),
        current_pairings=len(current_pairings),
        temp_c=temp_c,
        min_delay=min_delay,
        max_delay=max_delay,
        manual_x=manual_x,
    )
    warnings: list[str] = []
    print_candidate_summary(visual_events, audio_events)
    print("\nReview mode:")
    print("Press Enter to keep the suggested pairing, type an audio index to override, or 's' to skip.")

    current_by_visual = {event.visual_candidate_id: event for event in current_pairings}
    used_audio_ids: set[int] = set()
    updated_events: list[MatchedExplosionEvent] = []

    for visual_index, visual in enumerate(visual_events):
        progress(
            f"[review] Advancing to visual {visual_index + 1}/{len(visual_events)} "
            f"(candidate_id={visual.candidate_id}, t={visual.time_sec:.3f}s)."
        )
        current = current_by_visual.get(visual.candidate_id)
        default_label = "none"
        if current is not None:
            default_audio_index = next(
                (index for index, audio in enumerate(audio_events) if audio.candidate_id == current.audio_candidate_id),
                None,
            )
            default_label = str(default_audio_index) if default_audio_index is not None else "none"

        while True:
            response = input(
                f"Visual [{visual_index}] at {visual.time_sec:.3f}s -> audio index [{default_label}]? "
            ).strip()
            progress_kv("[review] Received pairing response", response=response, visual_candidate_id=visual.candidate_id)
            if response == "":
                if current is not None:
                    used_audio_ids.add(current.audio_candidate_id)
                    updated_events.append(current)
                    progress(
                        f"[review] Kept suggested pairing for visual candidate {visual.candidate_id} "
                        f"using audio candidate {current.audio_candidate_id}."
                    )
                else:
                    progress(
                        f"[review] No suggested pairing existed for visual candidate {visual.candidate_id}; "
                        "continuing without a match."
                    )
                break
            if response.lower() in {"s", "skip"}:
                progress(f"[review] Skipped visual candidate {visual.candidate_id}.")
                break
            try:
                audio_index = int(response)
            except ValueError:
                print("Please enter an integer audio index, Enter, or 's'.")
                continue
            if audio_index < 0 or audio_index >= len(audio_events):
                print("Audio index out of range.")
                continue

            chosen_audio = audio_events[audio_index]
            if chosen_audio.candidate_id in used_audio_ids and (
                current is None or chosen_audio.candidate_id != current.audio_candidate_id
            ):
                print("That audio candidate is already assigned. Choose another or skip.")
                continue

            if current is not None:
                used_audio_ids.discard(current.audio_candidate_id)
            used_audio_ids.add(chosen_audio.candidate_id)
            manual_event = build_matched_event(
                event_id=len(updated_events) + 1,
                visual_event=visual,
                audio_event=chosen_audio,
                temp_c=temp_c,
                min_delay=min_delay,
                max_delay=max_delay,
                extra_notes=["Manual pairing override."],
            )
            updated_events.append(manual_event)
            progress(
                f"[review] Assigned visual candidate {visual.candidate_id} "
                f"to audio candidate {chosen_audio.candidate_id}."
            )
            break

        progress(f"[review] Finished visual {visual_index + 1}/{len(visual_events)}.")

    for index, event in enumerate(updated_events, start=1):
        event.event_id = index
        if manual_x:
            matching_visual = next(
                item for item in visual_events if item.candidate_id == event.visual_candidate_id
            )
            progress(
                f"[review] Advancing to manual x-position review for event {event.event_id}/{len(updated_events)}."
            )
            should_select = input(
                f"Select/correct x-position for event {event.event_id}? "
                f"[current={event.x_position if event.x_position is not None else 'n/a'}] y/N: "
            ).strip().lower()
            progress_kv("[review] Received manual x response", response=should_select, event_id=event.event_id)
            if should_select in {"y", "yes"}:
                selected_position = _select_position(video_path, matching_visual.frame_index, event.event_id)
                if selected_position is None:
                    warnings.append(f"No x-position selected for event {event.event_id}.")
                else:
                    event.x_position = selected_position[0]
                    event.notes.append("x-position selected manually in review mode.")
            else:
                progress(f"[review] Kept existing x-position for event {event.event_id}.")

    return updated_events, warnings


def review_positions(
    video_path: str | Path,
    visual_events: list[VisualEventCandidate],
    events: list[MatchedExplosionEvent],
    manual_x: bool = False,
) -> tuple[list[MatchedExplosionEvent], list[str]]:
    """Let the user correct x positions for matched events by clicking the frame."""

    progress_kv(
        "[review] review_positions() called",
        video_path=str(video_path),
        event_count=len(events),
        manual_x=manual_x,
    )
    warnings: list[str] = []
    if not manual_x:
        return events, warnings

    visual_by_id = {item.candidate_id: item for item in visual_events}
    for event in events:
        matching_visual = visual_by_id.get(event.visual_candidate_id)
        if matching_visual is None:
            warnings.append(f"Could not find the source frame for event {event.event_id}.")
            continue

        prompt = (
            f"Select/correct x-position for event {event.event_id}? "
            f"[current={event.x_position if event.x_position is not None else 'n/a'}] y/N: "
        )

        should_select = input(prompt).strip().lower()
        progress_kv(
            "[review] Received manual point response",
            response=should_select,
            event_id=event.event_id,
            manual_x=manual_x,
        )
        if should_select not in {"y", "yes"}:
            continue

        selected_position = _select_position(video_path, matching_visual.frame_index, event.event_id)
        if selected_position is None:
            warnings.append(f"No manual point selected for event {event.event_id}.")
            continue
        event.x_position = selected_position[0]
        event.notes.append("x-position selected manually in review mode.")

    return events, warnings
