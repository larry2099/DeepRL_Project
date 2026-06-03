import win32gui
import win32con
import time

def enum_windows(hwnd, _):
    if win32gui.IsWindowVisible(hwnd):
        text = win32gui.GetWindowText(hwnd)
        if text:
            print(f"HWND: {hwnd}  Title: '{text}'")

print("Please open Geometry Dash manually (windowed mode) and then press Enter...")
input()
print("Listing all visible windows:")
win32gui.EnumWindows(enum_windows, None)