import glob as _glob
import os
import signal
import subprocess
import time

import cv2

import config
import harness
from vision import Vision, VisionState

import logging

logger = logging.getLogger(__name__)


class Event:
    def __init__(self, kind, body=None):
        self.kind = kind
        self.body = body


def _find_free_display(start: int = 99) -> int:
    """Return the first display number >= start with no existing X11 socket."""
    existing = {
        int(os.path.basename(p)[1:])
        for p in _glob.glob("/tmp/.X11-unix/X*")
        if os.path.basename(p)[1:].isdigit()
    }
    display = start
    while display in existing:
        display += 1
    return display


class LinuxGame:
    def __init__(
        self,
        vision: Vision | None = None,
        display: str | None = None,
        stream_port: int | None = None,
    ):
        logger.info("creating LinuxGame")

        self.display = display
        self.stream_port = (
            stream_port if stream_port is not None else config.FFMPEG_PORT
        )
        self.overlay_path = None

        self.wm_proc = None
        self.game_proc = None
        self.xvfb_proc = None
        self.ffmpeg_proc = None

        self.events = []
        self.vision = vision
        self.last_frame = None
        self.last_state: VisionState | None = None

        # Watchdog state
        self._last_frame_hash: float | None = None
        self._last_frame_change_time = time.perf_counter()
        self._last_alive_time = time.perf_counter()
        self._death_start_time: float | None = None

    def open(self):
        if self.display is None:
            self.display = f":{_find_free_display()}"
        self.overlay_path = f"/tmp/gdash_overlay_{self.display.lstrip(':')}.txt"

        logger.info(f"starting XVFB on {self.display}")
        self.xvfb_proc = subprocess.Popen(
            [
                "Xvfb",
                self.display,
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
        os.environ["DISPLAY"] = self.display

        logger.info("starting fluxbox")
        self.wm_proc = subprocess.Popen(
            ["fluxbox"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)

        logger.info(f"starting ffmpeg stream on port {self.stream_port}")
        # Ensure overlay file exists so drawtext can open it.
        with open(self.overlay_path, "w") as f:
            f.write("starting...")

        self.ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-f",
                "x11grab",
                "-r",
                "15",
                "-s",
                "800:600",
                "-i",
                f"{self.display}.0",
                "-vf",
                (
                    f"[in] drawtext=textfile={self.overlay_path}"
                    f":reload=1:x=10:y=10:fontfile={config.FONT_FILE}"
                    ":fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5 [out]"
                ),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "mpegts",
                f"tcp://0.0.0.0:{self.stream_port}?listen",
            ],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)

        logger.info("starting the game")
        self.game_proc = subprocess.Popen(
            [config.GAME_SCRIPT, config.GAME_PATH, self.display],
            env=os.environ,
            start_new_session=True,
        )
        time.sleep(5)

        logger.info("creating the harness")
        self.harness = harness.get_harness(self.display)
        logger.info("obtaining the window")
        self.window = self.harness.find_window(config.WINDOW_TITLE)
        if self.window is None:
            raise RuntimeError(
                f"could not find window '{config.WINDOW_TITLE}' on {self.display}"
            )

        if self.vision is None:
            logger.info("creating default vision module")
            self.vision = Vision("filters/death.png", exact_position=True)

        self._last_alive_time = time.perf_counter()
        self._death_start_time = None
        logger.info("welcome!")

    def update(self) -> VisionState:
        self.last_frame = self.harness.capture(self.window)
        self.last_state = self.vision.update(self.last_frame)

        now = time.perf_counter()

        # Frame-change watchdog
        frame_hash = float(cv2.mean(self.last_frame)[0])
        if self._last_frame_hash != frame_hash:
            self._last_frame_hash = frame_hash
            self._last_frame_change_time = now

        # Alive/death watchdog
        if self.last_state.is_dead:
            if self._death_start_time is None:
                self._death_start_time = now
            logger.info("died!")
            self.interact(delay=config.INPUT_FREQUENCY)
        else:
            self._death_start_time = None
            self._last_alive_time = now

        sleep_amt = config.INPUT_FREQUENCY
        for evt in self.events:
            if evt.kind == "hold_jump":
                self.harness.press_key(self.window, "up")
                time.sleep(config.INPUT_FREQUENCY)
                sleep_amt -= config.INPUT_FREQUENCY
            elif evt.kind == "release_jump":
                self.harness.release_key(self.window, "up")
                time.sleep(config.INPUT_FREQUENCY)
                sleep_amt -= config.INPUT_FREQUENCY
            elif evt.kind == "interact":
                key = evt.body["key"]
                delay = evt.body["delay"]
                self.harness.send_key(self.window, key)
                time.sleep(delay)
                sleep_amt -= delay
        self.events = []

        if sleep_amt > 0:
            time.sleep(sleep_amt)

        return self.last_state

    def hold_jump(self):
        self.events.append(Event("hold_jump"))

    def release_jump(self):
        self.events.append(Event("release_jump"))

    def interact(self, key="space", delay=1.0):
        self.events.append(Event("interact", {"key": key, "delay": delay}))

    def is_alive(self) -> bool:
        return self.game_proc is not None and self.game_proc.poll() is None

    def is_stuck(self, frame_timeout: float = 5.0, death_timeout: float = 30.0) -> bool:
        now = time.perf_counter()
        if now - self._last_frame_change_time > frame_timeout:
            return True
        if (
            self._death_start_time is not None
            and now - self._death_start_time > death_timeout
        ):
            return True
        return False

    def hard_restart(self) -> None:
        logger.warning("hard restart triggered")
        self.close()
        self.open()

    def close(self):
        logger.info("LinuxGame cleanup")

        logger.info("closing the processes")
        for proc in (self.game_proc, self.wm_proc, self.xvfb_proc, self.ffmpeg_proc):
            if proc is None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone

        for proc in (self.game_proc, self.wm_proc, self.xvfb_proc, self.ffmpeg_proc):
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

        self.wm_proc = None
        self.game_proc = None
        self.xvfb_proc = None
        self.ffmpeg_proc = None

        logger.info("good bye!")
