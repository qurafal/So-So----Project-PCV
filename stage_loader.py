import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChartNote:
    time: float
    side: str


@dataclass(frozen=True)
class StageData:
    window_title: str
    chart_offset_seconds: float
    chart_notes: list[ChartNote]
    song_path: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def resolve_relative_path(base_path: Path, relative_path: str) -> Path:
    return (base_path.parent / relative_path).resolve()


def load_chart_notes(chart_data: dict[str, Any]) -> list[ChartNote]:
    notes: list[ChartNote] = []

    for note_data in chart_data.get("notes", []):
        notes.append(
            ChartNote(
                time=float(note_data["time"]),
                side=str(note_data["side"]),
            )
        )

    notes.sort(key=lambda note: note.time)
    return notes


def load_stage(stage_path: str | Path) -> StageData:
    stage_path = Path(stage_path).resolve()
    stage_data = load_json(stage_path)

    chart_name = stage_data["song"]["chart_file"]
    chart_path = resolve_relative_path(stage_path, chart_name)
    chart_data = load_json(chart_path)
    chart_notes = load_chart_notes(chart_data)

    song_path = resolve_relative_path(stage_path, stage_data["song"]["file"])
    window_title = stage_data.get("camera", {}).get("window_title", "Rhythm Game")
    chart_offset_seconds = float(stage_data.get("song", {}).get("offset_seconds", 0.0))

    return StageData(
        window_title=window_title,
        chart_offset_seconds=chart_offset_seconds,
        chart_notes=chart_notes,
        song_path=song_path,
    )
