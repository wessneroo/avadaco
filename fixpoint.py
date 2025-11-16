import cv2
import numpy as np

vid = cv2.VideoCapture("videos\\example1.mov")

cv2.namedWindow("avadaco", cv2.WINDOW_NORMAL)
cv2.resizeWindow("avadaco", 1280, 720)


ret, frame1 = vid.read()
if not ret:
    print("Error: Failed to read video")
    vid.release()
    exit()
cv2.imshow("avadaco", frame1)
cv2.waitKey(1)
roi = cv2.selectROI("avadaco", frame1, showCrosshair=True)
#cv2.destroyWindow("avadaco")

x, y, w, h = roi
print("Fixpoint ROI:", x, y, w, h)
fp_crop = frame1[y:y+h, x:x+w]
fp_gray = cv2.cvtColor(fp_crop, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(
    fp_gray,
    maxCorners=20,
    qualityLevel=0.3,
    minDistance=3
)
if p0 is not None:
    p0_full = p0 + [[x, y]] # Adjust points to full frame coordinates
else:
    p0_full = None

prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

print("Number of feature points detected:", 0 if p0_full is None else len(p0_full))
print("Initial feature points (full frame coords):\n", p0_full)

vis = cv2.cvtColor(fp_gray, cv2.COLOR_GRAY2BGR)  # convert gray → color so circles show properly
if p0 is not None:
    for pt in p0:
        cx, cy = pt.ravel()
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
cv2.imshow("Feature Points in ROI", vis)
cv2.waitKey(0)
cv2.destroyWindow("Feature Points in ROI")

det_roi = cv2.selectROI("avadaco", frame1, showCrosshair=True)
dx, dy, dw, dh = det_roi
#cv2.destroyWindow("avadaco")

rel_dx = dx - x
rel_dy = dy - y
rel_dh = dh
rel_dw = dw



while True:
    ret, frame = vid.read()
    if not ret:
        break
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        p0_full.astype(np.float32),
        None
    )
    good_new = p1[status == 1]
    good_old = p0_full[status == 1]
    dx = np.mean(good_new[:, 0] - good_old[:, 0])
    dy = np.mean(good_new[:, 1] - good_old[:, 1])
    x += dx
    y += dy
    det_x = int(x + rel_dx)
    det_y = int(y + rel_dy)
    det_w = rel_dw
    det_h = rel_dh
    x_int, y_int = int(x), int(y)
    cv2.rectangle(
        frame,
        (x_int, y_int),
        (x_int + w, y_int + h),
        (0, 255, 0),
        2
    )
    cv2.rectangle(
        frame,
        (det_x, det_y),
        (det_x + det_w, det_y + det_h),
        (255, 0, 0),
        2
    )
    cv2.imshow("avadaco", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    prev_gray = curr_gray.copy()
    p0_full = good_new.reshape(-1, 1, 2)



