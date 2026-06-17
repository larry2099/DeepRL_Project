import os

HOME = os.environ.get("HOME")
assert HOME is not None
GAME_PATH = HOME + "/Games/GD/GeometryDash.exe"

WINDOW_TITLE = "Geometry Dash"
PWD = os.getcwd()
GAME_SCRIPT = PWD + "/game.sh"

INPUT_DURATION = 0.05
INPUT_FREQUENCY = 0.05
FFMPEG_PORT = 8080

FONT_FILE = "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf"
FFMPEG_PATH = "/usr/bin/ffmpeg"

JUMP_PENALTY = 0.2
DEATH_PENALTY = 100.0
RESOLUTION = (120, 160)
