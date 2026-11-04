from psychopy import prefs

prefs.hardware["audioLib"] = ["sounddevice"]

from psychopy import core, event, sound, visual

win = visual.Window(size=(800, 600), color="grey", units="height")
text = visual.TextStim(win, text="Press 1-4, or q to quit")
text.draw()
win.flip()

sound.Sound("../recordings_wav/afrikaans1.wav").play()
keys = event.waitKeys(keyList=["1", "2", "3", "4", "q"])
print(keys)

win.close()
core.quit()