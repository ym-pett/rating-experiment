from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from psychopy import prefs

PARTICIPANT = "mp"

# early rating results in neg rt
ALLOW_EARLY_RATING = True  # set False to ignore ratings until the audio has finished

# Must run before psychopy.sound is imported, or PsychoPy picks psychtoolbox.
prefs.hardware["audioLib"] = ["sounddevice"]  # pyright: ignore

from psychopy import core, event, sound, visual

from scripts.results_store import (
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


def poll_keys(keys: list[str], clock: core.Clock) -> tuple[str, float] | None:
    pressed = event.getKeys(keyList=keys, timeStamped=clock)
    return (pressed[0][0], pressed[0][1]) if pressed else None

def poll_keys_blocking(clock: core.Clock) -> tuple[str, float]:
    key, key_time = event.waitKeys(keyList=ALLOWED_KEYS, timeStamped=clock)[0]
    return key, key_time

def rate_trial(
    screen: Screen, snd: sound.Sound, done: int, total: int, allow_early: bool
) -> tuple[str, float] | None:
    fraction = done / total
    counter = f"{done + 1} / {total}"
    duration = snd.getDuration()
    listening = "Listening...  (rate any time with 1-4)" if allow_early else "Listening..."
    during_playback = ALLOWED_KEYS if allow_early else [QUIT_KEY]
    while True:
        event.clearEvents()
        screen.show(f"{counter}\n{listening}", fraction)
        clock = core.Clock()  # starts with playback; rt = key time - duration
        snd.play()
        response = None
        while response is None and clock.getTime() < duration:
            response = poll_keys(during_playback, clock)
            core.wait(0.005)
        snd.stop()  # cuts the audio short if a key was pressed early
        if response is None:
            event.clearEvents()  # discard keys pressed while the audio was playing
            screen.show(f"{counter}\nRate 1-4   |   R = replay   |   Esc = quit", fraction)
            response = poll_keys_blocking(clock)
        key, key_time = response
        if key == QUIT_KEY:
            return None
        if key != REPLAY_KEY:
            return key, key_time - duration


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
    allow_early: bool,
) -> None:
    todo = remaining_trials(conditions, load_results(results_path), session)
    total = int((conditions["session"] == session).sum())
    already_done = total - len(todo)
    start = datetime.now().astimezone()
    n_rated = 0

    missing = [f for f in todo["audio_file"] if not (audio_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files not found, e.g. {missing[0]}")
    
    for i, trial in todo.iterrows():
        snd = sound.Sound(str(audio_dir / trial["audio_file"]))
        outcome = rate_trial(screen, snd, already_done + int(i), total, allow_early)
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
    audio_dir: Path, conditions_path: Path, results_path: Path, session_log_path: Path, allow_early: bool
) -> None:
    conditions = pd.read_csv(conditions_path)
    session = next_session(conditions, load_results(results_path))
    if session is None:
        print("All sessions complete.")
        return
    screen = make_screen()
    run_session(screen, conditions, session, audio_dir, results_path, session_log_path, allow_early)
    screen.win.close()
    core.quit()


if __name__ == "__main__":
    main(
        audio_dir=Path("../recordings_wav/"),
        conditions_path=Path(f"../conditions/conditions_{PARTICIPANT}.csv"),
        results_path=Path(f"../responses/results_{PARTICIPANT}.csv"),
        session_log_path=Path(f"../responses/sessions_{PARTICIPANT}.csv"),
        allow_early=ALLOW_EARLY_RATING,
    )