import os

HOME = os.environ.get("HOME")
assert HOME is not None
GAME_PATH = HOME + "/Games/GD/GeometryDash.exe"

WINDOW_TITLE = "Geometry Dash"
PWD = os.getcwd()
GAME_SCRIPT = PWD + "/game.sh"

INPUT_DURATION = 0.05
INPUT_FREQUENCY = 0.1
