import subprocess
import time
import os
import signal
import config
import harness
from vision import Vision, VisionState

import logging

logger = logging.getLogger(__name__)


class Event:
    def __init__(self, kind, body=None):
        self.kind = kind
        self.body = body


class LinuxGame:
    def __init__(self, vision: Vision | None = None):
        logger.info("creating LinuxGame")
        self.wm_proc = None
        self.game_proc = None
        self.xvfb_proc = None
        self.events = []
        self.vision = vision
        self.last_frame = None
        self.last_state: VisionState | None = None

    def open(self):
        logger.info("starting XVFB")
        self.xvfb_proc = subprocess.Popen(
            [
                "Xvfb",
                ":99",
                "-screen",
                "0",
                "800x600x24",
                "+extension",
                "RANDR",
                "+extension",
                "XTEST",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(1)
        os.environ["DISPLAY"] = ":99"

        logger.info("starting fluxbox")
        self.wm_proc = subprocess.Popen(
            ["fluxbox"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)

        logger.info("starting the game")
        self.game_proc = subprocess.Popen(
            [config.GAME_SCRIPT, config.GAME_PATH],
            env=os.environ,
            start_new_session=True,
        )
        time.sleep(5)

        logger.info("creating the harness")
        self.harness = harness.get_harness(":99")
        logger.info("obtaining the window")
        self.window = self.harness.find_window(config.WINDOW_TITLE)
        assert self.window is not None

        if self.vision is None:
            logger.info("creating default vision module")
            self.vision = Vision("filters/death.png", exact_position=True)

        logger.info("welcome!")

    def update(self) -> VisionState:
        self.last_frame = self.harness.capture(self.window)
        self.last_state = self.vision.update(self.last_frame)

        if self.last_state.just_died:
            logger.info("died!")
            self.interact()

        for evt in self.events:
            if evt.kind == "hold_jump":
                self.harness.press_key(self.window, "up")
                time.sleep(config.INPUT_FREQUENCY)
            elif evt.kind == "release_jump":
                self.harness.release_key(self.window, "up")
                time.sleep(config.INPUT_FREQUENCY)
            elif evt.kind == "interact":
                self.harness.send_key(self.window, evt.body)
                time.sleep(1)
        self.events = []

        return self.last_state

    def hold_jump(self):
        self.events.append(Event("hold_jump"))

    def release_jump(self):
        self.events.append(Event("release_jump"))

    def interact(self, key="space"):
        self.events.append(Event("interact", key))

    def close(self):
        logger.info("LinuxGame cleanup")

        logger.info("closing the processes")
        for proc in (self.game_proc, self.wm_proc, self.xvfb_proc):
            if proc is None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone

        for proc in (self.game_proc, self.wm_proc, self.xvfb_proc):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass

        logger.info("good bye!")
