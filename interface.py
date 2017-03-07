import subprocess
        
def get_mouse_location():
    stdout, _ = subprocess.Popen('xdotool getmouselocation', shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    out = dict(entry.split(':') for entry in stdout.split(' '))

    return int(out['y']), int(out['x']), int(out['window'])

def click_relative(y, x, t, window):
    _, _ = subprocess.Popen('xdotool mousemove --window {0} {1} {2}'.format(window, x, y), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    _, _ = subprocess.Popen('xdotool click --window {0} {1}'.format(window, t), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    _, _ = subprocess.Popen('xdotool type --window {1} {0}'.format('c', window), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    return

def refocus(y, x, window):
    _, _ = subprocess.Popen('xdotool mousemove {0} {1}'.format(x, y), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    _, _ = subprocess.Popen('xdotool windowfocus {0}'.format(window), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

def focus(y, x, window):
    _, _ = subprocess.Popen('xdotool windowfocus {0}'.format(window), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

def press(string, window):
    stdout, _ = subprocess.Popen('xdotool getactivewindow', shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

    owindow = stdout.strip()

    _, _ = subprocess.Popen('xdotool windowactivate --sync {0} key {1} windowactivate {2}'.format(window, string, owindow), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()


class MouseKeyboard(object):
    def __init__(self, window):
        self.wid = window

    def click(self, (x, y), t = 1):
        yt, xt, wt = get_mouse_location()
        
        click_relative(int(y) + 22, int(x) + 3, t, self.wid)
        
        refocus(yt, xt, wt)

    def press(self, string):
        print 'HI'
        press(string, self.wid)
