from numpy import random
import game
import logging
import signal

logging.basicConfig(level=logging.INFO)
g = game.LinuxGame()

def on_sigint(sig, frame):
    _ = sig 
    _ = frame

    global g
    while True:
        s = input("choose action: [i]: interact [q]: quit: ")
        if s == "i":
            g.interact()
            return
        elif s == "q":
            raise Exception("exit")

signal.signal(signal.SIGINT, on_sigint)

try:
    g.open()
    g.interact()
    g.interact()

    while True:
        action = random.rand()
        if action < 0.1:
            g.hold_jump()
        elif action > 0.9:
            g.release_jump()

        g.update()

except Exception as e:
    logging.error(e)
    g.close()

