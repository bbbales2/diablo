import sys, pygame
import os
import re
import numpy
import bisect
import json
import pickle
import matplotlib.pyplot as plt
import traceback
import googlenet
import mahotas
import time
import argparse

pygame.init()

import subprocess
import gtk
import registration
#import tensorflow as tf

parser = argparse.ArgumentParser(description='Run a Diablo 2 bot')
#parser.add_argument('classifiersFile', help = 'File that has the trained classifiers')
parser.add_argument('--windowName', type = str, default = "Diablo II.exe", help = 'Window name for the running Diablo 2')

# These are some xtool helper functions

args = parser.parse_args()

def get_mouse_location():
    stdout, _ = subprocess.Popen('xdotool getmouselocation', shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    out = dict(entry.split(':') for entry in stdout.split(' '))

    return int(out['y']), int(out['x']), int(out['window'])

def click_relative(y, x, t, window):
    _, _ = subprocess.Popen('xdotool mousemove --window {0} {1} {2}'.format(window, x, y), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    _, _ = subprocess.Popen('xdotool click --window {0} {1}'.format(window, t), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    return

def refocus(y, x, window):
    _, _ = subprocess.Popen('xdotool mousemove {0} {1}'.format(x, y), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    _, _ = subprocess.Popen('xdotool windowfocus {0}'.format(window), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

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

def d2click(y, x, t):
    yt, xt, wt = get_mouse_location()

    click_relative(y, x, t, wid)

    refocus(yt, xt, wt)

window = gtk.gdk.window_foreign_new(wid)

W, H = window.get_size()

screengrab = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, W, H)

def get_screen():
    screengrab.get_from_drawable(window,
                                 gtk.gdk.colormap_get_system(),
                                 0, 0, 0, 0,
                                 W,
                                 H)

    data = numpy.frombuffer(screengrab.get_pixels(), numpy.uint8)
    data = data.reshape((H, screengrab.get_rowstride()))
    data = data[:, :W * 3]
    data = data.reshape((H, W, 3))

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

    def handle(self, event):
        ctrl_pressed = pygame.key.get_mods() & (pygame.KMOD_RCTRL | pygame.KMOD_LCTRL)

        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_f:
                pass
        #try:
        #    self.selected.handle(event, self)
        #except Exception as e:
        #    traceback.print_exc()

    def draw(self):
        tmp = time.time()

        screen = get_screen()
        surf = pygame.surfarray.make_surface(numpy.rollaxis(screen, 1, 0))

        #l, offset = registration.mle_loc(screen)
        l, offset, reg = registration.mle_loc(screen)

        reg -= reg.min()
        reg /= reg.max()

        toblit = (plt.cm.viridis(numpy.kron(reg, numpy.ones((8, 8))))[:, :, :3] * 255).astype('uint8')
        toblit2 = pygame.surfarray.make_surface(numpy.rollaxis(toblit, 1, 0))
        toblit2.set_alpha(63)
        
        pygame.draw.circle(surf, [255, 0, 0], (int(offset[1]), int(offset[0])), 5)
        self.screen.blit(surf.convert(), (0, 0))
        self.screen.blit(toblit2, (0, 0))

        processTimes.append(time.time() - tmp)

        if len(processTimes) > 10:
            processTimes.pop(0)
        
        lines = []
        lines.append("FPS: {0:.2f} / s".format(1.0 / numpy.mean(processTimes)))
        lines.append("")
        
        lines.append("Frame estimate: {0}".format(l))
        lines.append("Offset est.: {0}".format(offset.astype('int')))

        lines.append("")
        #lines.append(self.msg)

        for i, line in enumerate(lines):
            label = font.render(line, 1, (255, 255, 255))
            g.screen.blit(label, (g.W, 15 * i))
 
g = Global(screen, W, H)

while 1:
    for event in pygame.event.get():
        g.handle(event)

    clock.tick(50)

    screen.fill((0, 0, 0))

    g.draw()

    pygame.display.flip()
