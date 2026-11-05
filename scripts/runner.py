from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from psychopy import prefs

# Must run before psychopy.sound is imported, or PsychoPy picks psychtoolbox.
prefs.hardware["audioLib"] = ["sounddevice"]  # pyright: ignore

from psychopy import core, event, sound, visual  # noqa: E402

from scripts.results_store import (  # noqa: E402
    append_result,
    append_session,
    load_results,
    next_session,
    remaining_trials,
)

RATING_LABELS = {
    "1": "Native-like",
    "2": "Slight accent",
    "3": "Slight accent but more pronounced than 2",
    "4": "Recognisably accented but no impact on intelligibility",
    "5": "Recognisably accented with some impacts on intelligibility",
    "6": "Accent noticeably affects intelligibility",
}
REPLAY_KEY = "r"
QUIT_KEY = "escape"
ALLOWED_KEYS = [*RATING_LABELS, REPLAY_KEY, QUIT_KEY]
BAR_HEIGHT = 0.04
BAR_Y = 0.98


@dataclass
class Screen:
    win: visual.Window
    track: visual.Rect
    bar: visual.Rect
    status: visual.TextStim
    labels: visual.TextStim

    def show(self, status: str, fraction: float) -> None:
        self.bar.size = (2 * fraction, BAR_HEIGHT)
        self.bar.pos = (-1 + fraction, BAR_Y)
        self.status.text = status
        for stim in (self.track, self.bar, self.status, self.labels):
            stim.draw()
        self.win.flip()


def make_screen() -> Screen:
    win = visual.Window(size=(1100, 750), color="black", units="height")
    label_text = "\n\n".join(f"{key}   {text}" for key, text in RATING_LABELS.items())
    return Screen(
        win=win,
        track=visual.Rect(win, units="norm", size=(2, BAR_HEIGHT), pos=(0, BAR_Y),
                          fillColor="#333333", lineColor=None),
        bar=visual.Rect(win, units="norm", size=(0.001, BAR_HEIGHT), pos=(-1, BAR_Y),
                        fillColor="#4a90d9", lineColor=None),
        status=visual.TextStim(win, pos=(0, 0.3), height=0.05, color="white"),
        labels=visual.TextStim(win, text=label_text, pos=(0, -0.05), height=0.045,
                               wrapWidth=1.2, alignText="left", color="white"),
    )


def rate_trial(
    screen: Screen, snd: sound.Sound, done: int, total: int
) -> tuple[str, float] | None:
    fraction = done / total
    counter = f"{done + 1} / {total}"
    while True:
        screen.show(f"{counter}\nListening...", fraction)
        snd.play()
        core.wait(snd.getDuration())
        event.clearEvents()  # discard keys pressed while the audio was playing
        clock = core.Clock()  # rt is measured from here: the end of playback
        screen.show(f"{counter}\nRate 1-6   |   R = replay   |   Esc = quit", fraction)
        key, rt = event.waitKeys(keyList=ALLOWED_KEYS, timeStamped=clock)[0]
        if key == QUIT_KEY:
            return None
        if key != REPLAY_KEY:
            return key, rt


def make_result_row(trial: pd.Series, key: str, rt: float) -> dict[str, object]:
    return {
        "participant": trial["participant"],
        "round": trial["round"],
        "session": trial["session"],
        "audio_file": trial["audio_file"],
        "rating": int(key),
        "rt": round(rt, 3),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_session(
    screen: Screen,
    conditions: pd.DataFrame,
    session: int,
    audio_dir: Path,
    results_path: Path,
    session_log_path: Path,
) -> None:
    todo = remaining_trials(conditions, load_results(results_path), session)
    total = int((conditions["session"] == session).sum())
    already_done = total - len(todo)
    start = datetime.now().astimezone()
    n_rated = 0
    for i, trial in todo.iterrows():
        snd = sound.Sound(str(audio_dir / trial["audio_file"]))
        outcome = rate_trial(screen, snd, already_done + int(i), total)
        if outcome is None:
            break
        append_result(results_path, make_result_row(trial, *outcome))
        n_rated += 1
    end = datetime.now().astimezone()
    append_session(
        session_log_path,
        {
            "participant": todo.iloc[0]["participant"],
            "round": todo.iloc[0]["round"],
            "session": session,
            "date": start.date().isoformat(),
            "start_time": start.strftime("%H:%M:%S"),
            "end_time": end.strftime("%H:%M:%S"),
            "n_rated": n_rated,
        },
    )
    if n_rated == len(todo):
        screen.show("Session complete. Press any key.", 1.0)
        event.waitKeys()


def main(
    audio_dir: Path, conditions_path: Path, results_path: Path, session_log_path: Path
) -> None:
    conditions = pd.read_csv(conditions_path)
    session = next_session(conditions, load_results(results_path))
    if session is None:
        print("All sessions complete.")
        return
    screen = make_screen()
    run_session(screen, conditions, session, audio_dir, results_path, session_log_path)
    screen.win.close()
    core.quit()


if __name__ == "__main__":
    main(
        audio_dir=Path("../recordings_wav/test/"),
        conditions_path=Path("../conditions/minitest2.csv"),
        results_path=Path("../responses/results_test2.csv"),
        session_log_path=Path("../responses/sessions_test2.csv"),
    )