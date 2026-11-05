from pathlib import Path

import pandas as pd
import pytest

from scripts.build_conditions_file import build_conditions, save_conditions

FILES = [f"speaker{i}.wav" for i in range(2138)]


@pytest.fixture
def conditions() -> pd.DataFrame:
    return build_conditions(FILES, participant="test", n_sessions=8, seed=0)


def test_every_file_appears_exactly_once(conditions: pd.DataFrame) -> None:
    assert conditions["audio_file"].is_unique == True 


def test_session_sizes_differ_by_at_most_one(conditions: pd.DataFrame) -> None:
    # checking np.array_split works 
    sizes = conditions.groupby("session").size()
    assert sizes.max() - sizes.min() == 1


def test_positions_run_from_one_within_each_session(conditions: pd.DataFrame) -> None:
    for session, group in conditions.groupby("session"):
        assert group["position"].tolist() == list(range(1, len(group) +1))


def test_same_seed_gives_same_conditions() -> None:
    first = build_conditions(FILES, participant="test", n_sessions=8, seed=0)
    second = build_conditions(FILES, participant="test", n_sessions=8, seed=0)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_order() -> None:
    first = build_conditions(FILES, participant="test", n_sessions=8, seed=1)
    second = build_conditions(FILES, participant="test", n_sessions=8, seed=0)
    assert first["audio_file"].equals(second["audio_file"]) == False


def test_save_refuses_to_overwrite(conditions: pd.DataFrame, tmp_path: Path) -> None:
    # tmp_path is pytest inbuilt, gives fresh, empty folder
    path = tmp_path / "conditions.csv"
    save_conditions(conditions, path)

    with pytest.raises(FileExistsError):
        save_conditions(conditions, path)