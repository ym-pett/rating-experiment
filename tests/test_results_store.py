from pathlib import Path

import pandas as pd
import pytest

from scripts.build_conditions_file import build_conditions
from scripts.results_store import (
    RESULT_COLUMNS,
    append_result,
    load_results,
    next_session,
    remaining_trials,
)

FILES = [f"speaker{i}.wav" for i in range(10)]


@pytest.fixture
def conditions() -> pd.DataFrame:
    return build_conditions(FILES, participant="test", n_sessions=2, seed=0)


@pytest.fixture
def no_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def rated(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[["participant", "round", "session", "audio_file"]].assign(
        rating=3, rt=1.0, timestamp="2026-11-05T10:00:00"
    )


def make_row(audio_file: str) -> dict[str, object]:
    return {
        "participant": "test",
        "round": "rating",
        "session": 1,
        "audio_file": audio_file,
        "rating": 2,
        "rt": 1.5,
        "timestamp": "t",
    }


def test_load_results_missing_file_gives_empty_frame(tmp_path: Path) -> None:
    results = load_results(tmp_path / "nope.csv")
    assert results.empty
    assert list(results.columns) == RESULT_COLUMNS


def test_append_result_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    append_result(path, make_row("a.wav"))
    append_result(path, make_row("b.wav"))
    assert load_results(path)["audio_file"].tolist() == ["a.wav", "b.wav"]


def test_append_result_rejects_wrong_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        append_result(tmp_path / "results.csv", {"participant": "test"})


def test_remaining_trials_without_results_is_whole_session(
    conditions: pd.DataFrame, no_results: pd.DataFrame
) -> None:
    todo = remaining_trials(conditions, no_results, session=1)
    assert todo["position"].tolist() == [1, 2, 3, 4, 5]


def test_remaining_trials_skips_rated_files(
    conditions: pd.DataFrame,
) -> None:
    session_one = conditions[conditions["session"] == 1]
    results = rated(session_one.head(2))
    todo = remaining_trials(conditions, results, session=1)
    assert todo["position"].tolist() == [3, 4, 5]


def test_remaining_trials_does_not_mix_up_rounds(conditions: pd.DataFrame) -> None:
    rerating = conditions.assign(round="rerating")
    results = rated(conditions)  # everything rated, but in round "rating"
    assert len(remaining_trials(rerating, results, session=1)) == 5


def test_next_session_is_first_when_nothing_rated(
    conditions: pd.DataFrame, no_results: pd.DataFrame
) -> None:
    assert next_session(conditions, no_results) == 1


def test_next_session_stays_on_partly_done_session(conditions: pd.DataFrame) -> None:
    session_one = conditions[conditions["session"] == 1]
    assert next_session(conditions, rated(session_one.head(2))) == 1


def test_next_session_moves_on_when_session_finished(conditions: pd.DataFrame) -> None:
    session_one = conditions[conditions["session"] == 1]
    assert next_session(conditions, rated(session_one)) == 2


def test_next_session_is_none_when_all_done(conditions: pd.DataFrame) -> None:
    assert next_session(conditions, rated(conditions)) is None