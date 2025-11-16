import cv2

vid = cv2.VideoCapture("videos\\example1.mov")

ret, frame1 = vid.read()
if not ret:
    print("Error: Failed to read video")
    vid.release()
    exit()
roi = cv2.selectROI("Select ROI", frame1, showCrosshair=True)
cv2.destroyWindow("Select ROI")

x, y, w, h = roi
print("Fixpoint ROI:", x, y, w, h)


# while True:
#     ret, frame = vid.read()
#     if not ret:
#         break
#     cv2.imshow("Video Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
