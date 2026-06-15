#!/usr/bin/env python3
"""
Cross-platform game harness for RL screen-capture + input injection.

Dependencies
------------
All platforms:
    pip install mss numpy Pillow
Windows:
    pip install pywin32
Linux:
    pip install python-xlib

Key names
---------
``send_key`` and ``send_key_focused`` accept either:
* A single character string: ``" "``, ``"a"``, ``"A"``.
* A named key string (case-insensitive on Windows, tries a few
  capitalisations on Linux): ``"left"``, ``"right"``, ``"space"``,
  ``"return"``, ``"shift"``, ``"ctrl"``, ``"escape"``, etc.
* A raw integer (platform-specific): Windows VK code or X11 keycode.

Caveats
-------
- ``send_key`` tries to inject directly into the window's event queue.
  Some games ignore this (DirectInput/RawInput on Windows, or windows
  that don't select for KeyPress on X11). Use ``send_key_focused`` as
  a fallback in those cases.
- ``capture`` uses ``mss`` to read the desktop at the window's screen
  coordinates. If another window overlaps the target, you will capture
  the overlap. Run the game unobstructed or borderless-fullscreen.
- On Windows true-exclusive-fullscreen, DWM may not composite the game
  to the screen DC and ``mss`` can return black. Borderless windowed
  mode is safer for capture.
"""

from __future__ import annotations

import platform
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from mss import mss
import config


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class GameHarness(ABC):
    """Abstract harness. Concrete implementations are OS-specific below."""

    @abstractmethod
    def find_window(self, title_pattern: str) -> Optional[Any]:
        """Return a native window handle whose title matches *title_pattern*."""
        raise NotImplementedError

    @abstractmethod
    def get_window_geometry(self, handle: Any) -> Dict[str, int]:
        """Return ``{"left": x, "top": y, "width": w, "height": h}``."""
        raise NotImplementedError

    _KEY_HOLD_DURATION = 0.05

    @abstractmethod
    def press_key(self, handle: Any, key: Union[str, int]) -> None:
        """Send a key-press event directly to the window."""
        raise NotImplementedError

    @abstractmethod
    def release_key(self, handle: Any, key: Union[str, int]) -> None:
        """Send a key-release event directly to the window."""
        raise NotImplementedError

    def send_key(self, handle: Any, key: Union[str, int]) -> None:
        """Press and release *key* directly to the window (no focus change)."""
        self.press_key(handle, key)
        time.sleep(self._KEY_HOLD_DURATION)
        self.release_key(handle, key)

    @abstractmethod
    def send_key_focused(self, handle: Any, key: Union[str, int]) -> None:
        """Bring window to foreground/focus, then inject the key globally."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared: capture via mss (desktop grab + crop)
    # ------------------------------------------------------------------

    def capture(self, handle: Any) -> np.ndarray:
        """Grab the window and return an ``(H, W, 3)`` uint8 RGB array."""
        geo = self.get_window_geometry(handle)
        with mss() as sct:
            shot = sct.grab(geo)
            # ``shot.bgra`` is raw BGRA bytes.
            img = np.frombuffer(shot.bgra, dtype=np.uint8)
            img = img.reshape((shot.height, shot.width, 4))
            # Drop alpha channel and flip BGR -> RGB.
            return img[..., :3][..., ::-1]


# ---------------------------------------------------------------------------
# Windows implementation
# ---------------------------------------------------------------------------


class WindowsHarness(GameHarness):
    def __init__(self) -> None:
        import win32api
        import win32con
        import win32gui

        self._win32api = win32api
        self._win32con = win32con
        self._win32gui = win32gui

    # ------------------------------------------------------------------

    def find_window(self, title_pattern: str) -> Optional[int]:
        matches: list[int] = []

        def _enum_cb(hwnd: int, _: Any) -> None:
            if self._win32gui.IsWindowVisible(hwnd):
                title = self._win32gui.GetWindowText(hwnd)
                if re.search(title_pattern, title, re.IGNORECASE):
                    matches.append(hwnd)

        self._win32gui.EnumWindows(_enum_cb, None)
        return matches[0] if matches else None

    # ------------------------------------------------------------------

    def get_window_geometry(self, handle: int) -> Dict[str, int]:
        # ``GetClientRect`` gives the render area (no title bar / borders).
        left, top, right, bottom = self._win32gui.GetClientRect(handle)
        # Convert client-area (0,0) to absolute screen coordinates.
        x, y = self._win32gui.ClientToScreen(handle, (left, top))
        return {
            "left": x,
            "top": y,
            "width": right - left,
            "height": bottom - top,
        }

    # ------------------------------------------------------------------

    def _resolve_vk(self, key: Union[str, int]) -> int:
        """Turn *key* into a Windows virtual-key code."""
        if isinstance(key, int):
            return key

        if len(key) == 1:
            vk = self._win32api.VkKeyScan(key)
            if vk == -1:
                raise ValueError(f"Cannot map character {key!r} to a VK code")
            return vk & 0xFF

        # Named key: e.g. "left" -> VK_LEFT, "space" -> VK_SPACE, …
        attr = f"VK_{key.upper()}"
        if hasattr(self._win32con, attr):
            return getattr(self._win32con, attr)

        raise ValueError(
            f"Unknown key {key!r}. Pass a single character, a VK name "
            f"(e.g. 'LEFT', 'SPACE'), or an int VK code."
        )

    def _pynput_key(self, key: Union[str, int]):
        """Turn *key* into something pynput understands."""
        from pynput.keyboard import Key

        if isinstance(key, int):
            # pynput can't consume raw VK codes easily; convert to char if printable
            if 0x20 <= key <= 0x7E:
                return chr(key)
            raise ValueError(
                f"Integer VK code {key} is not supported by the focused "
                f"fallback on Windows. Use a named key string instead."
            )

        if len(key) == 1:
            return key

        attr = key.lower()
        if hasattr(Key, attr):
            return getattr(Key, attr)

        raise ValueError(f"Unknown named key {key!r} for pynput fallback")

    # ------------------------------------------------------------------

    def press_key(self, handle: int, key: Union[str, int]) -> None:
        """PostMessage WM_KEYDOWN directly to *handle*."""
        vk_code = self._resolve_vk(key)
        scan_code = self._win32api.MapVirtualKey(vk_code, 0)
        lparam = (scan_code << 16) | 1
        self._win32api.PostMessage(handle, self._win32con.WM_KEYDOWN, vk_code, lparam)

    def release_key(self, handle: int, key: Union[str, int]) -> None:
        """PostMessage WM_KEYUP directly to *handle*."""
        vk_code = self._resolve_vk(key)
        scan_code = self._win32api.MapVirtualKey(vk_code, 0)
        lparam = ((scan_code << 16) | 1) | (1 << 30) | (1 << 31)
        self._win32api.PostMessage(handle, self._win32con.WM_KEYUP, vk_code, lparam)

    # ------------------------------------------------------------------

    def send_key_focused(self, handle: int, key: Union[str, int]) -> None:
        """Set foreground window, then inject globally with pynput."""
        import ctypes

        ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
        self._win32gui.SetForegroundWindow(handle)
        time.sleep(0.15)  # Give the WM time to activate the window

        from pynput.keyboard import Controller

        pkey = self._pynput_key(key)
        ctrl = Controller()
        ctrl.press(pkey)
        time.sleep(0.05)
        ctrl.release(pkey)


# ---------------------------------------------------------------------------
# Linux / X11 implementation
# ---------------------------------------------------------------------------


class LinuxHarness(GameHarness):
    _KEY_HOLD_DURATION = config.INPUT_DURATION

    def __init__(self, display: Optional[str] = None) -> None:
        import Xlib.display
        import Xlib.XK

        self._display = Xlib.display.Display(display)
        self._XK = Xlib.XK
        self._root = self._display.screen().root

    # ------------------------------------------------------------------

    def find_window(self, title_pattern: str) -> Optional[Any]:
        def _recurse(window) -> Optional[Any]:
            try:
                name = window.get_wm_name()
            except Exception:
                name = None
            if name and re.search(title_pattern, name, re.IGNORECASE):
                return window
            for child in window.query_tree().children:
                found = _recurse(child)
                if found is not None:
                    return found
            return None

        return _recurse(self._root)

    # ------------------------------------------------------------------

    def get_window_geometry(self, handle: Any) -> Dict[str, int]:
        geo = handle.get_geometry()
        abs_pos = handle.translate_coords(self._root, 0, 0)
        return {
            "left": abs_pos.x,
            "top": abs_pos.y,
            "width": geo.width,
            "height": geo.height,
        }

    # ------------------------------------------------------------------

    def _resolve_keysym(self, key: Union[str, int]) -> int:
        """Turn *key* into an X keysym."""
        if isinstance(key, int):
            # Assume caller passed a raw keysym value.
            return key

        keysym = self._XK.string_to_keysym(key)
        if keysym == 0:
            keysym = self._XK.string_to_keysym(key.capitalize())
        if keysym == 0:
            keysym = self._XK.string_to_keysym(key.upper())
        if keysym == 0:
            raise ValueError(
                f"Unknown key {key!r}. Pass a single character, an X keysym "
                f"name (e.g. 'Left', 'space', 'Return'), or an int keysym."
            )
        return keysym

    # ------------------------------------------------------------------

    def _xevent(self, handle: Any, keycode: int, pressed: bool):
        from Xlib import X
        from Xlib.protocol.event import KeyPress, KeyRelease

        cls = KeyPress if pressed else KeyRelease
        return cls(
            time=X.CurrentTime,
            root=self._root,
            window=handle,
            same_screen=1,
            child=X.NONE,
            root_x=0,
            root_y=0,
            event_x=0,
            event_y=0,
            state=0,
            detail=keycode,
        )

    def press_key(self, handle: Any, key: Union[str, int]) -> None:
        """Send KeyPress directly via XSendEvent."""
        from Xlib import X

        keycode = self._display.keysym_to_keycode(self._resolve_keysym(key))
        handle.send_event(
            self._xevent(handle, keycode, True),
            propagate=False,
            event_mask=X.KeyPressMask,
        )
        self._display.sync()

    def release_key(self, handle: Any, key: Union[str, int]) -> None:
        """Send KeyRelease directly via XSendEvent."""
        from Xlib import X

        keycode = self._display.keysym_to_keycode(self._resolve_keysym(key))
        handle.send_event(
            self._xevent(handle, keycode, False),
            propagate=False,
            event_mask=X.KeyReleaseMask,
        )
        self._display.sync()

    # ------------------------------------------------------------------

    def send_key_focused(self, handle: Any, key: Union[str, int]) -> None:
        """XSetInputFocus + XTEST fake_input (like xdotool)."""
        from Xlib import X
        from Xlib.ext.xtest import fake_input

        self._display.set_input_focus(handle, X.RevertToParent, X.CurrentTime)
        self._display.sync()
        time.sleep(0.05)

        keysym = self._resolve_keysym(key)
        keycode = self._display.keysym_to_keycode(keysym)

        fake_input(self._display, X.KeyPress, keycode)
        self._display.sync()
        time.sleep(0.05)
        fake_input(self._display, X.KeyRelease, keycode)
        self._display.sync()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_harness(display: Optional[str] = None) -> GameHarness:
    """Return an OS-specific harness instance.

    On Linux you can optionally pass the *display* string (e.g. ``":99"``).
    On Windows the argument is ignored.
    """
    system = platform.system()
    if system == "Windows":
        return WindowsHarness()
    if system == "Linux":
        return LinuxHarness(display=display)
    raise RuntimeError(f"Unsupported platform: {system}")
