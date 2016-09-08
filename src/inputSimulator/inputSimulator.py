import sys
sys.path.append("..\segmentation")

import numpy as np
import win32api, win32con
from screenSegmentation import screenSegmentation
import time

class InputSimulator():
    
    def __init__(self):
        sS = screenSegmentation()
        self.ox, self.oy = sS.getGameOrigin()

    def leftClick(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        
    def mousePos(self, x, y):
        win32api.SetCursorPos( (self.ox + x, self.oy + y) )
        
    def clickPlay(self):
        self.mousePos(410,252)
        self.leftClick()
        self.mousePos(600,400)
        
    def clickTryAgain(self):
        self.mousePos(460,412)
        self.leftClick()
        self.mousePos(600,400)
        
    def space(self):
        win32api.keybd_event(win32con.VK_SPACE, 0)
        time.sleep(.05)
        win32api.keybd_event(win32con.VK_SPACE, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def left(self):
        win32api.keybd_event(win32con.VK_LEFT, 0)
        time.sleep(.05)
        win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def right(self):
        win32api.keybd_event(win32con.VK_RIGHT, 0)
        time.sleep(.05)
        win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def up(self):
        win32api.keybd_event(win32con.VK_UP, 0)
        time.sleep(.05)
        win32api.keybd_event(win32con.VK_UP, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def down(self):
        win32api.keybd_event(win32con.VK_DOWN, 0)
        time.sleep(.05)
        win32api.keybd_event(win32con.VK_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def c(self):
        win32api.keybd_event(67, 0)
        time.sleep(.05)
        win32api.keybd_event(67, 0, win32con.KEYEVENTF_KEYUP, 0)
        
    def retry(self):
        self.space()
        
    def enterInitials(self):
        win32api.keybd_event(68, 0)
        time.sleep(.05)
        win32api.keybd_event(68, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(76, 0)
        time.sleep(.05)
        win32api.keybd_event(76, 0, win32con.KEYEVENTF_KEYUP, 0)
        self.mousePos(403,409)
        self.leftClick()
        self.mousePos(600,400)

def main():
    iS = InputSimulator()
    iS.clickPlay()

if __name__ == '__main__':
    main()