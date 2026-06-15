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
        s = input("choose action: [i]: interact [j]: jump [q]: quit: ")
        if s == "i":
            g.interact()
            return
        elif s == "j":
            g.jump()
            return
        elif s == "q":
            raise Exception("exit")

signal.signal(signal.SIGINT, on_sigint)

try:
    g.open()
    g.interact()
    g.interact()

    while True:
        g.jump()
        g.update()

except Exception as e:
    logging.error(e)
    g.close()

