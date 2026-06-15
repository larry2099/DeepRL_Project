import game
import logging

logging.basicConfig(level=logging.INFO)

g = game.LinuxGame()
try:
    g.open()
    while True:
        pass
except:
    g.close()

