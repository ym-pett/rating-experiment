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


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path)


def append_result(path: Path, row: dict[str, object]) -> None:
    if set(row) != set(RESULT_COLUMNS):
        raise ValueError(f"expected keys {RESULT_COLUMNS}, got {sorted(row)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


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