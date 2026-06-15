import subprocess
import time
import os
import config
import harness

import logging
logger = logging.getLogger(__name__)

class LinuxGame:
    def __init__(self):
        logger.info("creating LinuxGame")
        self.wm_proc = None
        self.game_proc = None
        self.xvfb_proc = None

    def open(self):
        logger.info("starting XVFB")
        self.xvfb_proc = subprocess.Popen(
            [
                "Xvfb",
                ":99",
                "-screen", "0", "800x600x24",
                "+extension", "RANDR",
                "+extension", "XTEST",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        os.environ["DISPLAY"] = ":99"

        logger.info("starting fluxbox")
        self.wm_proc = subprocess.Popen(
            ["fluxbox"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        logger.info("starting the game")
        self.game_proc = subprocess.Popen(
            [config.GAME_SCRIPT, config.GAME_PATH],
            env=os.environ,
        )
        time.sleep(5)

        logger.info("creating the harness")
        self.harness = harness.get_harness(":99")
        logger.info("obtaining the window")
        self.window = self.harness.find_window(config.WINDOW_TITLE)
        assert(self.window is not None)
        logger.info("welcome!")

    def close(self):
        logger.info("LinuxGame cleanup")

        logger.info("closing the processes")
        if self.game_proc is not None: self.game_proc.terminate()
        if self.wm_proc is not None: self.wm_proc.terminate()
        if self.xvfb_proc is not None: self.xvfb_proc.terminate()

        logger.info("good bye!")
