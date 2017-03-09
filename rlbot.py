import sys, pygame
import os
import re
import numpy
import bisect
import json
import pickle
import matplotlib.pyplot as plt
import traceback
import time
import argparse
import threading
import tempfile
import gzip

pygame.init()

import subprocess
import gtk
import rl
# These are some xtool helper functions
import interface

parser = argparse.ArgumentParser(description='Run a Diablo 2 bot')
parser.add_argument('botFile', help = 'File to save bot in')
parser.add_argument('--windowName', type = str, default = "Diablo II.exe", help = 'Window name for the running Diablo 2')

args = parser.parse_args()

# Locate a running Diablo II.exe to latch on to

sp = subprocess.Popen('xdotool search --name "{0}"'.format(args.windowName), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

stdout, stderr = sp.communicate()

wids = []
for line in stdout.split('\n'):
    line = line.strip()

    if len(line) > 0:
        try:
            line = int(line)
            wids.append(line)
        except:
            print "Failed to parse {0} as line".format(line)
            pass

if len(wids) == 0:
    print """No Diablo 2 clients found, please verify Diablo 2 is running

If Diablo 2 is running, try providing the window name as a '--windowName argument'"

If this doesn't work make sure xdotool can find the window with something like:
    xdotool search --name \"Diablo II.exe\"

This should return some xids."""
    exit(0)
elif len(wids) == 1:
    wid = wids[0]
elif len(wids) > 1:
    print "Multiple Diablo 2 clients found, select which one to attach to"
    print "This code is untested -- might not work. Why do you have multiple Diablo 2s running?"

    for i, wid in enumerate(wids):
        print "[{0}] {1}".format(i, wid)

    i = input('Select client: ')

    try:
        i = int(i)
    except:
        print "Client id must be a number"

        traceback.print_exc()

        exit(-1)

    if i >= len(wids) or i < 0:
        print "Client id must be within 0 to {0}".format(len(wids) - 1)

    wid = wids[i]

print "Diablo 2 exe found"

mk = interface.MouseKeyboard(wid)

window = gtk.gdk.window_foreign_new(wid)

W_, H_ = window.get_size()
W = 640
H = 480

screengrab = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, W_, H_)

def get_screen():
    screengrab.get_from_drawable(window,
                                 gtk.gdk.colormap_get_system(),
                                 0, 0, 0, 0,
                                 W_,
                                 H_)

    data = numpy.frombuffer(screengrab.get_pixels(), numpy.uint8)
    data = data.reshape((H_, screengrab.get_rowstride()))
    data = data[:, :W_ * 3]
    data = data.reshape((H_, W_, 3))
    data = data[22 : H + 22, 3 : W + 3]

    return data

# Load up the neural network

screen = pygame.display.set_mode((W + 200, H))

clock = pygame.time.Clock()
pygame.key.set_repeat(200, 25)
font = pygame.font.SysFont("monospace", 15)

msg = ""

processTimes = []

class Global(object):
    def __init__(self, screen = None, W = None, H = None):
        self.screen = screen
        self.selected = None
        self.font = font
        self.W = W
        self.H = H
        self.ticks = 0
        self.mk = mk
        self.resetBot()
        self.tick()

    def getScreen(self):
        return get_screen()

    def resetGame(self):
        self.ticks = 0
        mk.press("Escape")
        time.sleep(0.1)
        mk.click((282, 199)) # Save and exit
        time.sleep(1.0)
        mk.click((403, 303)) # Single Player
        time.sleep(0.25)
        mk.click((690, 543)) # Okay (character select)
        time.sleep(4.0)
        self.resetBot()
        self.tick()

    def resetBot(self):
        self.bot = rl.Bot()

    def saveBot(self):
        with open(args.botFile, 'w') as f:
            pickle.dump(self.bot, f)

    def tick(self):
        try:
            self.bot.tick(self)
            self.ticks += 1
        except Exception as e:
            traceback.print_exc()
            print "Error in tick: {0}".format(e)
            
        if self.ticks == 10:
            with tempfile.NamedTemporaryFile(dir = '/home/bbales2/diablo/trials', delete = False) as f:
                gzf = gzip.GzipFile(mode = 'wb', fileobj = f)
                pickle.dump(self.bot.recording, gzf)
                gzf.close()

            self.resetGame()
        else:
            threading.Timer(1.0, self.tick).start()

    def handle(self, event):
        ctrl_pressed = pygame.key.get_mods() & (pygame.KMOD_RCTRL | pygame.KMOD_LCTRL)

        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            mk.click(pos)

            """1070, 384
1451, 807, menu
1327, 805, character

1352, 583, save and exit
1473, 687, single player
1760, 927, Okay"""

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_c:
                mk.click((257, 421))
            elif event.key == pygame.K_r:
                self.bot_reset()

        #for bot in self.bots:
        self.bot.handle(event, self)
        #try:
        #    self.selected.handle(event, self)
        #except Exception as e:
        #    traceback.print_exc()

    def draw(self):
        tmp = time.time()

        screen = get_screen()
        surf = pygame.surfarray.make_surface(numpy.rollaxis(screen, 1, 0))

        self.screen.blit(surf.convert(), (0, 0))

        botlines = []
        botlines.extend(self.bot.draw(screen, self))

        processTimes.append(time.time() - tmp)

        if len(processTimes) > 10:
            processTimes.pop(0)
        
        lines = []
        lines.append("FPS: {0:.2f} / s".format(1.0 / numpy.mean(processTimes)))
        lines.append("")

        lines.extend(botlines)

        lines.append("")
        #lines.append(self.msg)

        for i, line in enumerate(lines):
            label = font.render(line, 1, (255, 255, 255))
            g.screen.blit(label, (g.W, 15 * i))
 
g = Global(screen, W, H)

while 1:
    for event in pygame.event.get():
        g.handle(event)

    clock.tick(67)

    screen.fill((0, 0, 0))

    g.draw()

    pygame.display.flip()
