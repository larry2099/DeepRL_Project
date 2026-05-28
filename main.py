#!/usr/bin/env python3
import os
import subprocess
import time
import random
import asyncio

print("Starting Xvfb...")
xvfb_proc = subprocess.Popen(
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
time.sleep(1)  # Give Xvfb time to initialise

# All X clients spawned from here on connect to our headless server
os.environ["DISPLAY"] = ":99"

print("Launching fluxbox...")
wm_proc = subprocess.Popen(
  ["fluxbox"],
  env=os.environ,
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL,
)
time.sleep(2)

print("Launching game...")
home = os.environ.get("HOME")
assert home is not None
game_path = home + "/Games/GD/GeometryDash.exe"
demo_proc = subprocess.Popen(
    ["./game.sh",  game_path],
    env=os.environ,
)
time.sleep(5)  # Let the window map

async def captureDisplay():
    frame_time = 1 / 30
    cur_frame = 0
    with mss() as sct:
        # Grab the full 800x600 screen explicitly
        while True:
            screenshot = sct.grab({"top": 0, "left": 0, "width": 800, "height": 600})
            mid_x, mid_y = 400, 300
            r, g, b = screenshot.pixel(mid_x, mid_y)
            print(f"[frame {cur_frame}]: RGB({r}, {g}, {b})")
            cur_frame += 1
            await asyncio.sleep(frame_time)

async def doInput():
    keyboard = Controller()
    keyboard.press(" ")
    await asyncio.sleep(0.1)
    keyboard.release(" ")
    await asyncio.sleep(1)
    keyboard.press(Key.right)
    await asyncio.sleep(0.1)
    keyboard.release(Key.right)
    await asyncio.sleep(1)
    keyboard.press(" ")
    await asyncio.sleep(0.1)
    keyboard.release(" ")
    await asyncio.sleep(1)

    while True:
        await asyncio.sleep(random.random())
        keyboard.press(" ")
        await asyncio.sleep(0.1)
        keyboard.release(" ")

async def main():
    await asyncio.gather(doInput(), captureDisplay())

    await asyncio.sleep(0.5)
    demo_proc.terminate()
    wm_proc.terminate()
    xvfb_proc.terminate()
    print("Done.")

asyncio.run(main())
