import cv2
import numpy as np
import tkinter as tk

def nothing(x):
    pass


def mouse_crop(self, event, x, y, flags, param):
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
        self.cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if self.cropping:
            self.x_end, self.y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        self.x_end, self.y_end = x, y
        self.cropping = False # cropping is finished

# 91 50 48.6

root = tk.Tk()
root.withdraw()

screen_width = self.root.winfo_screenwidth()
screen_height = self.root.winfo_screenheight()

cap = cv2.VideoCapture("Cycle-Camera 1 (C11141).avi")

# Read first frame.
#ret, frame = cap.read()

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.namedWindow("Trackbars_u")
cv2.createTrackbar("U - H", "Trackbars_u", 0, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars_u", 0, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars_u", 0, 255, nothing)
cv2.namedWindow("select area, use q when happy")
cv2.setMouseCallback("select area, use q when happy", mouse_crop)
_, frame = cap.read()
cv2.resizeWindow('select area, use q when happy', frame.shape[1], frame.shape[0])
x_pos = round(screen_width / 2) - round(frame.shape[1] / 2)
y_pos = round(screen_height / 2) - round(frame.shape[0] / 2)
cv2.moveWindow("select area, use q when happy", x_pos, y_pos)
hsv_low = np.zeros((250, 500, 3), np.uint8)
hsv_high = np.zeros((250, 500, 3), np.uint8)
while True:

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars_u")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars_u")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars_u")
    hsv_low[:] = (l_h, l_s, l_v)
    hsv_high[:] = (u_h, u_s, u_v)
    imbgr_l = cv2.cvtColor(hsv_low, cv2.COLOR_HSV2BGR)
    imbgr_u = cv2.cvtColor(hsv_high, cv2.COLOR_HSV2BGR)
    cv2.imshow("Trackbars_u", imbgr_u)
    cv2.imshow("Trackbars", imbgr_l)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
