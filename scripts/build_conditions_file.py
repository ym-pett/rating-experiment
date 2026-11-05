import secrets
from pathlib import Path

import numpy as np
import pandas as pd

audio_dir = Path("../recordings_wav/")
conditions_file = Path("../conditions/test.csv")

def list_audio_files(audio_dir: Path) -> list[str]:
    return sorted(str(p.relative_to(audio_dir)) for p in audio_dir.rglob("*.wav"))


def build_conditions(
    files: list[str], participant: str, n_sessions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(files)
    chunks = np.array_split(shuffled, n_sessions)
    return pd.DataFrame(
        [
            {
                "participant": participant,
                "round": "rating",
                "session": session,
                "position": position,
                "audio_file": audio_file,
                "seed": seed
            }
            for session, chunk in enumerate(chunks, start=1)
            for position, audio_file in enumerate(chunk.tolist(), start=1)
        ]
    )


def save_conditions(conditions: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to reshuffle")
    path.parent.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(path, index=False)


def main(audio_dir: Path, output_path: Path, n_sessions: int ,participant: str, seed: int) -> None:
    audio_list = list_audio_files(audio_dir)
    conditions = build_conditions(audio_list, participant=participant,n_sessions=n_sessions, seed=seed)
    save_conditions(conditions, Path(conditions_file))



if __name__ == "__main__":
    seed = secrets.randbits(32)
    main(audio_dir=audio_dir, output_path=conditions_file, n_sessions=8, participant="mp", seed=seed)