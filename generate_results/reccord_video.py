import cv2
import numpy as np

if __name__ == "__main__":
    video_path = r"C:\Users\amede\Videos\Captures\Rerun Viewer 2024-06-07 18-41-27.mp4"
    new_path = r"C:\Users\amede\Videos\Captures\Rerun Viewer 2024-06-07 18-41-27.avi"
    cap = cv2.VideoCapture(video_path)
    size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    reccorder = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 110, (int(size[0]), int(size[1])))
    count = 0
    while count < 1870 / 2:
        ret, frame = cap.read()
        if not ret:
            break
        # print mouse position on the image
        # def on_mouse(event, x, y, flags, params):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         print('Mouse Position:', x, y)
        # cv2.setMouseCallback("frame", on_mouse)
        # blur circle around the mouse
        cv2.circle(frame, (720, 200), 100, (128, 128, 128), -1)
        # put text with white background
        x, y = 5, 5
        delta = 20
        delta_x = 10
        color = (255, 255, 255)
        shapes = frame
        text = "Generated using pyorerun, Puchaud & Begon (2024)"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(
            shapes,
            (x + delta_x, y + delta + 4),
            (x + delta_x + 2 + w, y + delta - h),
            color,
            -1,
        )
        cv2.putText(
            shapes,
            text,
            (x + delta_x, y + delta),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # mask_img = np.zeros(frame.shape, dtype='uint8')
        # mask_img = frame.copy()

        # cv2.circle(mask_img, (720, 200), 100, (255, 255, 255), -1)
        #
        # img_all_blurred = cv2.GaussianBlur(mask_img, (71, 71), 0)
        # img = np.where(mask_img > 0, img_all_blurred, frame)
        cv2.imshow("frame", frame)
        # cv2.imshow("circle", img)

        reccorder.write(frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    reccorder.release()
