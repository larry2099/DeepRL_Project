import time
import win32gui
import win32con
import pygetwindow as gw

class GameInterface:
    def __init__(self, target_title='Geometry Dash'):
        self.hwnd = None
        self.target_title = target_title
        self._find_window()

    def _find_window(self):
        windows = gw.getWindowsWithTitle(self.target_title)
        if windows:
            self.hwnd = windows[0]._hWnd
            print(f"✅ Attached to existing game window: '{self.target_title}'")
            self._activate_window()
        else:
            raise RuntimeError(f"❌ Could not find window with title '{self.target_title}'. Please open Geometry Dash manually (windowed mode) and run again.")

    def _activate_window(self):
        if self.hwnd:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.2)

    def get_window_rect(self):
        if self.hwnd:
            return win32gui.GetWindowRect(self.hwnd)
        return None

    def close(self):
        # Do not close the game – leave it open for the user
        pass