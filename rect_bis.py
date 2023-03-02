"""
Author: Stefan Palm
stefan.palm@hotmail.se
code is intended for learning
"""

import cv2
import tkinter as tk
from tkinter import filedialog
import os.path
from pathlib import Path
import itertools
import time
import numpy as np


class MarkerSet:
    """
    This class is used to store the marker information
    """
    def __init__(self, nb_markers: int, marker_names: list, image_idx: int):
        """
        init markers class with number of markers, names and image index

        Parameters
        ----------
        nb_markers : int
            number of markers
        marker_names : list
            list of names for the markers
        image_idx : list
            index of the image where the marker set is located
        """
        self.nb_markers = nb_markers
        self.image_idx = image_idx
        self.marker_names = marker_names
        self.pos = np.zeros((2, nb_markers, 1))
        self.speed = np.zeros((2, nb_markers, 1))
        self.marker_set_model = None
        self.markers_idx_in_image = []
        self.estimated_area = []
        self.next_pos = np.zeros((2, nb_markers, 1))

    @staticmethod
    def compute_speed(pos, pos_old, dt=1):
        """
        Compute the speed of the markers
        """
        return (pos - pos_old) / dt

    @staticmethod
    def compute_next_position(speed, pos, dt=1):
        """
        Compute the next position of the markers
        """
        return pos + speed * dt

    def update_speed(self):
        for i in range(2):
            self.speed[i, :] = self.compute_speed(self.pos[i, :, -1], self.pos[i, :, -2])

    def update_next_position(self):
        """
        Update the next position of the markers
        """
        next_pos=[]
        for i in range(2):
            # next_pos.append(np.concatenate((self.next_pos[i, :, :], self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]), axis=1))
            self.next_pos[i, :] = self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]


class Cropper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        #catching some info from root for later
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Select the file to work on, need to use withdraw to get rid of dialogue window
        self.videofile = "Cycle-Camera 1 (C11141).avi"
        # self.videofile = "viridis_101.png"

        n_sub = 3
        pos_dic = {}
        pos_dic["pos1"] = np.ndarray((2, 6))
        pos_dic["pos2"] = np.ndarray((2, 2))
        pos_dic["pos3"] = np.ndarray((2, 5))
        pos_dic["pos1"][0, :] = [37, 64, 54, 152, 140, 121]
        pos_dic["pos1"][1, :] = [65, 128, 197, 164, 102, 57]
        pos_dic["pos2"][0, :] = [72, 125]# [30, 74]
        pos_dic["pos2"][1, :] = [72, 125]
        pos_dic["pos3"][0, :] = [41, 90, 99, 68, 90]
        pos_dic["pos3"][1, :] = [85, 67, 87, 31, 19]
        markers =  {}
        markers["larm"] = []
        markers["epic_l"] = []

        find_bounds = False
        self.cap = cv2.VideoCapture(self.videofile)

        # Check if video opened successfully - TBD: nice exit
        if (self.cap.isOpened() == False):
          print("Unable to read file!")
          exit()

        while(self.cap.isOpened()):
            # Capture first frame and exit loop
            self.ret, self.frame = self.cap.read()
            break
        # release the video capture object, make sure we start from frame 1 next time
        self.cap.release()
        markers_arm = MarkerSet(nb_markers=2, marker_names=["arm_l", "epic_l"], image_idx=1)
        markers_arm.estimated_area.append([0, self.frame.shape[0], 0, self.frame.shape[1]])  # [x_start, x_end, y_start, y_end]
        markers_arm.estimated_area.append([0, self.frame.shape[0], 0, self.frame.shape[1]])
        # markers_arm.pos[0] = [30, 74]
        # markers_arm.pos[1] = [72, 125]
        def nothing(x):
            pass
        if find_bounds:
            cv2.namedWindow("Trackbars")
            cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
            cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
            cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
            cv2.namedWindow("Trackbars_u")
            cv2.createTrackbar("U - H", "Trackbars_u", 0, 255, nothing)
            cv2.createTrackbar("U - S", "Trackbars_u", 0, 255, nothing)
            cv2.createTrackbar("U - V", "Trackbars_u", 0, 255, nothing)
            hsv_low = np.zeros((250, 500, 3), np.uint8)
            hsv_high = np.zeros((250, 500, 3), np.uint8)

        #prep some stuff for cropping
        self.cropping = False
        self.oriImage = self.frame.copy()
        #prep a window, and place it at the center of the screen

        self.x_pos = round(self.screen_width/2) - round(self.frame.shape[1]/2)
        self.y_pos = round(self.screen_height/2) - round(self.frame.shape[0]/2)
        # self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop = [0], [0], [round(self.frame.shape[1])], [round(self.frame.shape[0])]
        self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop = [732, 900, 1017], [400, 541, 480], [908, 1040, 1207], [641, 721, 653]
        self.cropped = []
        for i in range(n_sub):
            self.cropped.append(self.oriImage[self.y_start_crop[i]:self.y_end_crop[i], self.x_start_crop[i]:self.x_end_crop[i]])
        print(self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop)

        # Now let us crop the video with same cropping parameters
        self.cap = cv2.VideoCapture(self.videofile)


        # self.frame[:, :, 0] = 0 * self.copy[:, :, 0] + 0 *  self.copy[:, :, 1] -1 * self.copy[:, :, 2]
        # self.frame[:, :, 1] = 0 * self.copy[:, :, 0] + 0 *  self.copy[:, :, 1] + 0 * self.copy[:, :, 2]
        # self.frame[:, :, 2] = 0 * self.copy[:, :, 0] + 0 *  self.copy[:, :, 1] - 1 * self.copy[:, :, 2]
        count_frame = 0
        #read frame by frame
        while(True):
            self.ret, self.frame = self.cap.read()
            if find_bounds:
                l_h = cv2.getTrackbarPos("L - H", "Trackbars")
                l_s = cv2.getTrackbarPos("L - S", "Trackbars")
                l_v = cv2.getTrackbarPos("L - V", "Trackbars")
                u_h = cv2.getTrackbarPos("U - H", "Trackbars_u")
                u_s = cv2.getTrackbarPos("U - S", "Trackbars_u")
                u_v = cv2.getTrackbarPos("U - V", "Trackbars_u")
                hsv_low[:] = (l_h, l_s, l_v)
                hsv_high[:] = (u_h, u_s, u_v)
                cv2.imshow("Trackbars_u", hsv_high)
                cv2.imshow("Trackbars", hsv_low)
            else:
                l_h, l_s, l_v, u_h, u_s, u_v = 101, 20, 0, 108, 255, 114
                l_h, l_s, l_v, u_h, u_s, u_v = 26, 73, 0, 43, 184, 121
            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])

            if self.ret:
                mask = []
                for i in range(n_sub):
                    cx, cy = [], []
                    self.cropped[i] = self.frame[self.y_start_crop[i]:self.y_end_crop[i], self.x_start_crop[i]:self.x_end_crop[i]]
                    hsv = cv2.cvtColor(self.cropped[i], cv2.COLOR_BGR2HSV)
                    mask.append(cv2.inRange(hsv, lower_blue, upper_blue))
                    result = cv2.bitwise_and(self.cropped[i], self.cropped[i], mask=mask[i])
                    imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
                    ret, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
                    if i == 1:
                        area_x = []
                        area_x = [area_x + markers_arm.estimated_area[x][:2] for x in range(len(markers_arm.estimated_area))]
                        area_x = list(itertools.chain.from_iterable(area_x))
                        area_x = [int(min(area_x)), int(max(area_x))]
                        area_y = []
                        area_y = [area_y + markers_arm.estimated_area[x][2:] for x in
                               range(len(markers_arm.estimated_area))]
                        area_y = list(itertools.chain.from_iterable(area_y))
                        area_y = [int(min(area_y)), int(max(area_y))]
                    else:
                        area_x, area_y = [0, self.cropped[i].shape[0]], [0, self.cropped[i].shape[1]]

                    contours, hierarchy = cv2.findContours(image=thresh[area_y[0]:area_y[1], area_x[0]:area_x[1]], mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
                    imgray = cv2.cvtColor(self.cropped[i], cv2.COLOR_BGR2GRAY)
                    contours_final, cX, cY = [], [], []
                    contour_threshold = 4

                    # TODO: Optim label marker using predefined model for hand and shoulder (find the index and the model position to minimize the least square error)
                    #  Once the first frame found use it as the model. always use the last frame as the model for the next one.
                    contours = self.agglomerative_cluster(list(contours), threshold_distance=20)

                    for c, contour in enumerate(contours):
                        if cv2.contourArea(contour) > contour_threshold:
                            contours_final.append(contour)
                            M = cv2.moments(contour)
                            cX = int(M["m10"] / M["m00"]) + area_x[0]
                            cY = int(M["m01"] / M["m00"]) + area_y[0]
                            # cv2.drawContours(image=imgray, contours=contour, contourIdx=-1, color=(255, 0, 0), thickness=2,
                            #                      lineType=cv2.LINE_AA)
                            if i == 1:
                                cx.append(cX)
                                cy.append(cY)
                    if i != 1:
                        for j in range(pos_dic[list(pos_dic.keys())[i]].shape[1]):
                            cx.append(int(pos_dic[list(pos_dic.keys())[i]][0, j]))
                            cy.append(int(pos_dic[list(pos_dic.keys())[i]][1, j]))
                    # check position for each marker depend on the others for second image check the y position
                    if i == 1:
                        if count_frame == 0:
                            markers_arm.pos[0, :, 0] = [cx[cy.index(min(cy))],
                                                        cx[cy.index(max(cy))]]
                            markers_arm.pos[1, :, 0] = [cy[cy.index(min(cy))],
                                                        cy[cy.index(max(cy))]]
                            markers_arm.speed = np.zeros((2, markers_arm.nb_markers))

                        else:
                            pos_x = np.append(markers_arm.pos[0, :, :], np.array([np.nan, np.nan])[:, np.newaxis], axis=1)
                            pos_y = np.append(markers_arm.pos[1, :, :], np.array([np.nan, np.nan])[:, np.newaxis], axis=1)
                            markers_arm.pos = np.concatenate((pos_x, pos_y), axis=0).reshape(2, markers_arm.nb_markers, count_frame + 1)
                            for c in range(len(cx)):
                                for p in range(markers_arm.nb_markers):
                                    if markers_arm.estimated_area[p][0] <= cx[c] <= markers_arm.estimated_area[p][1] and \
                                    markers_arm.estimated_area[p][2] <= cy[c] <= markers_arm.estimated_area[p][3]:
                                        markers_arm.pos[0, p, count_frame] = cx[c]
                                        markers_arm.pos[1, p, count_frame] = cy[c]
                            idx = np.argwhere(np.isnan(markers_arm.pos[:, :, count_frame]))
                            if idx.size != 0:
                                for id in idx:
                                    # markers_arm.pos[id[0], id[1], count_frame] = markers_arm.compute_next_position(markers_arm.speed[id[0], id[1]], markers_arm.pos[id[0], id[1], -2])
                                    markers_arm.pos[id[0], id[1], count_frame] = markers_arm.pos[id[0], id[1], count_frame-1]
                            markers_arm.update_speed()
                        markers_arm.update_next_position()
                        for m in range(markers_arm.nb_markers):
                            markers_arm.estimated_area[m] = [markers_arm.next_pos[0, m, 0] - 15,
                                                             markers_arm.next_pos[0, m, 0] + 15,
                                                             markers_arm.next_pos[1, m, 0] - 15,
                                                             markers_arm.next_pos[1, m, 0] + 15]
                            markers_arm.estimated_area[m] = list(np.array(markers_arm.estimated_area[m]).clip(min=0, max=max(self.frame.shape)))

                        for n, name in enumerate(markers_arm.marker_names):
                            if np.isfinite(markers_arm.pos[0, n, -1]):
                                cv2.circle(imgray, (int(markers_arm.pos[0, n, -1]), int(markers_arm.pos[1, n, -1])), 3, (255, 255, 255), -1)
                                cv2.putText(imgray, name, (int(markers_arm.pos[0, n, -1]- 20), int(markers_arm.pos[1, n, -1] - 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                        # cv2.rectangle(imgray, (cx[idx_arm_l]-20, cy[idx_arm_l]-20), (cx[idx_arm_l]+20, cy[idx_epic]+20), (255, 0, 0),
                        #               2)
                        # cv2.namedWindow(f'None approximation {i}', cv2.WINDOW_NORMAL)
                        # cv2.imshow(f'None approximation {i}', imgray)
                        if i == 1:
                            area_x = []
                            area_x = [area_x + markers_arm.estimated_area[x][:2] for x in
                                      range(len(markers_arm.estimated_area))]
                            area_x = list(itertools.chain.from_iterable(area_x))
                            area_x = [int(min(area_x)), int(max(area_x))]
                            area_y = []
                            area_y = [area_y + markers_arm.estimated_area[x][2:] for x in
                                      range(len(markers_arm.estimated_area))]
                            area_y = list(itertools.chain.from_iterable(area_y))
                            area_y = [int(min(area_y)), int(max(area_y))]
                        cv2.rectangle(imgray, (area_x[0], area_y[0]),
                                      (area_x[1], area_y[1]), (255, 0, 0), 2)
                        cv2.namedWindow(f'None approximation {i}', cv2.WINDOW_NORMAL)
                        cv2.imshow(f'None approximation {i}', imgray)
                    else:
                        for j in range(pos_dic[list(pos_dic.keys())[i]].shape[1]):
                            cv2.circle(imgray, (cx[j], cy[j]), 3, (255, 255, 255), -1)
                            cv2.putText(imgray, "center", (cx[j]- 20, cy[j] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    cv2.namedWindow(f'HSV video{i}', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'HSV video{i}', hsv)
                    cv2.namedWindow(f'original', cv2.WINDOW_NORMAL)
                    cv2.imshow('original', self.cropped[i])
                    # cv2.namedWindow(f'copy', cv2.WINDOW_NORMAL)
                    # cv2.imshow('copy', self.copy)
                    # cv2.imshow(f"result{i}", result)
                time.sleep(0.005)
                # Press Q on keyboard to stop recording early
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    res = input("Would you like to select other cropping zone ? (y/n)")
                    if res == "y":
                        cv2.destroyAllWindows()
                        self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop = [], [], [], []
                        self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop = self.select_cropping(n_sub)
                        cv2.namedWindow("Trackbars")
                        cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
                        cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
                        cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
                        cv2.namedWindow("Trackbars_u")
                        cv2.createTrackbar("U - H", "Trackbars_u", 0, 255, nothing)
                        cv2.createTrackbar("U - S", "Trackbars_u", 0, 255, nothing)
                        cv2.createTrackbar("U - V", "Trackbars_u", 0, 255, nothing)
                    else:
                        pass
                count_frame += 1
        # Break the loop when done

            else:
                self.cap = cv2.VideoCapture(self.videofile)
                self.ret, self.frame = self.cap.read()
                count_frame = 0
                markers_arm = MarkerSet(nb_markers=2, marker_names=["arm_l", "epic_l"], image_idx=1)
                markers_arm.estimated_area.append(
                    [0, self.frame.shape[0], 0, self.frame.shape[1]])  # [x_start, x_end, y_start, y_end]
                markers_arm.estimated_area.append([0, self.frame.shape[0], 0, self.frame.shape[1]])


        # When everything done, release the video capture and video write objects
        self.cap.release()

        # Make sure all windows are closed
        cv2.destroyAllWindows()

        # Leave a message

    @staticmethod
    def calculate_contour_distance(contour1, contour2):
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        c_x1 = x1 + w1 / 2
        c_y1 = y1 + h1 / 2

        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        c_x2 = x2 + w2 / 2
        c_y2 = y2 + h2 / 2

        return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)

    @staticmethod
    def merge_contours(contour1, contour2):
        return np.concatenate((contour1, contour2), axis=0)

    def agglomerative_cluster(self, contours, threshold_distance=5.0):
        current_contours = contours
        while len(current_contours) > 1:
            min_distance = None
            min_coordinate = None

            for x in range(len(current_contours) - 1):
                for y in range(x + 1, len(current_contours)):
                    distance = self.calculate_contour_distance(current_contours[x], current_contours[y])
                    if min_distance is None:
                        min_distance = distance
                        min_coordinate = (x, y)
                    elif distance < min_distance:
                        min_distance = distance
                        min_coordinate = (x, y)

            if min_distance < threshold_distance:
                index1, index2 = min_coordinate
                current_contours[index1] = self.merge_contours(current_contours[index1], current_contours[index2])
                del current_contours[index2]
            else:
                break

        return current_contours

    def mouse_crop(self, event, x, y, flags, param):
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.x_start_crop.append(x)
            self.y_start_crop.append(y)
            self.cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                if self.x_end != x and self.y_end != y:
                    self.x_end, self.y_end = x, y
        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            self.x_end, self.y_end = x, y
            self.x_end_crop.append(x)
            self.y_end_crop.append(y)
            self.cropping = False # cropping is finished

    def select_cropping(self, n_sub):
        self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop = [], [], [], []
        for it in range(n_sub):
            self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
            cv2.namedWindow("select area, use q when happy" + f"{it}")
            cv2.setMouseCallback("select area, use q when happy" + f"{it}", self.mouse_crop)
            cv2.resizeWindow('select area, use q when happy' + f"{it}", self.frame.shape[1], self.frame.shape[0])
            while True:
                i = self.frame.copy()
                if not self.cropping:
                    cv2.imshow("select area, use q when happy" + f"{it}", self.frame)
                    if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                        cv2.rectangle(i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                        cv2.imshow("select area, use q when happy" + f"{it}", i)

                elif self.cropping:
                    cv2.rectangle(i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use q when happy" + f"{it}", i)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            self.cropping = True
            self.cropped.append(self.oriImage[self.y_start_crop[it]:self.y_end_crop[it], self.x_start_crop[it]:self.x_end_crop[it]])
            it += 1
        return self.x_start_crop, self.y_start_crop, self.x_end_crop, self.y_end_crop

if __name__ == "__main__":
    app = Cropper()