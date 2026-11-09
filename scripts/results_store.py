from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "participant",
    "round",
    "session",
    "audio_file",
    "rating",
    "rt",
    "timestamp",
]

SESSION_COLUMNS = [
    "participant",
    "round",
    "session",
    "date",
    "start_time",
    "end_time",
    "n_rated",
]


def _append_row(path: Path, row: dict[str, object], columns: list[str]) -> None:
    if set(row) != set(columns):
        raise ValueError(f"expected keys {columns}, got {sorted(row)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=columns).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )

def append_session(path: Path, row: dict[str, object]) -> None:
    _append_row(path, row, SESSION_COLUMNS)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path)


def append_result(path: Path, row: dict[str, object]) -> None:
    _append_row(path, row, RESULT_COLUMNS)


def remaining_trials(
    conditions: pd.DataFrame, results: pd.DataFrame, session: int
) -> pd.DataFrame:
    in_session = conditions[conditions["session"] == session]
    done = results[["round", "audio_file"]].drop_duplicates()
    merged = in_session.merge(done, on=["round", "audio_file"], how="left", indicator=True)
    todo = merged[merged["_merge"] == "left_only"]
    return todo.drop(columns="_merge").sort_values("position").reset_index(drop=True)


def next_session(conditions: pd.DataFrame, results: pd.DataFrame) -> int | None:
    sessions = sorted(conditions["session"].unique())
    return next(
        (int(s) for s in sessions if not remaining_trials(conditions, results, s).empty),
        None,
    )