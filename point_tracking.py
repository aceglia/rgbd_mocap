import cv2
import sys
# import randint
import numpy as np
import random
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[4]
if tracker_type == 'BOOSTING':
    tracker = (cv2.legacy.TrackerBoosting_create)
if tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create
if tracker_type == 'KCF':
    tracker = (cv2.legacy.TrackerKCF_create)
if tracker_type == 'TLD':
    tracker = (cv2.legacy.TrackerTLD_create)
if tracker_type == 'MEDIANFLOW':
    tracker = (cv2.legacy.TrackerMedianFlow_create)
if tracker_type == 'GOTURN':
    tracker = (cv2.TrackerGOTURN_create)
if tracker_type == 'MOSSE':
    tracker = (cv2.legacy.TrackerMOSSE_create)
if tracker_type == "CSRT":
    tracker = (cv2.TrackerCSRT_create)


# Create a video capture object to read videos
cap = cv2.VideoCapture("Cycle-Camera 1 (C11141).avi")

# Read first frame.
ret, frame = cap.read()

frame_height, frame_width = frame.shape[:2]
output = cv2.VideoWriter(f'{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width, frame_height), True)
# quit if unable to read the video file
if not ret:
    print('Cannot read video file')
    sys.exit()
# Select boxes
bboxes = []
colors = []
p1, p2 = [], []
# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break
print('Selected bounding boxes {}'.format(bboxes))
# Specify the tracker type
# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()
# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(tracker(), frame, bbox)
# Process video and track objects

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)
    # draw tracked objects

    for i, newbox in enumerate(boxes):
        p1.append((int(newbox[0]), int(newbox[1])))
        p2.append((int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3])))
        # if p1[]

        cv2.rectangle(frame, p1[-1], p2[-1], colors[i], 2, 1)
    # show frame
    cv2.imshow('MultiTracker', frame)
    output.write(frame)
    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
# When everything done, release the video capture and video write objects
cap.release()
output.release()

# Closes all the frames
cv2.destroyAllWindows()
