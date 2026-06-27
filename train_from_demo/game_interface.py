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
            return True
        else:
            print(f"❌ Could not find window with title '{self.target_title}'.")
            self.hwnd = None
            return False

    def _activate_window(self):
        if self.hwnd:
            try:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self.hwnd)
            except Exception as e:
                print(f"⚠️ Window activation failed: {e}")
            time.sleep(0.3)

    def move_window(self, x, y, width, height):
        if self.hwnd:
            try:
                win32gui.MoveWindow(self.hwnd, x, y, width, height, True)
            except:
                pass

    def get_window_rect(self):
        if self.hwnd:
            try:
                return win32gui.GetWindowRect(self.hwnd)
            except:
                return None
        return None

    def close(self):
        pass