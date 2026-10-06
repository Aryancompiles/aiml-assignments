import cv2
import numpy as np
import os

print("=" * 65)
print("ASSIGNMENT 9: OBJECT DETECTION FROM CCTV FOOTAGE")
print("=" * 65)

# ---------------------------------------------------------
# Step 1: Generate a simulated CCTV video
# ---------------------------------------------------------

video_name = "simulated_cctv.mp4"
output_name = "annotated_cctv.mp4"
frame_image = "detected_frame.png"

width, height = 640, 360
fps = 20
total_frames = 200

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    video_name,
    fourcc,
    fps,
    (width, height)
)

for frame_no in range(total_frames):

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background
    frame[:] = (40, 40, 40)

    # Moving object 1
    x1 = 50 + (frame_no * 2) % 500
    cv2.rectangle(
        frame,
        (x1, 100),
        (x1 + 70, 160),
        (180, 180, 180),
        -1
    )

    # Moving object 2
    x2 = 500 - (frame_no * 2) % 450
    cv2.circle(
        frame,
        (x2, 250),
        30,
        (120, 120, 120),
        -1
    )

    # CCTV-style timestamp
    cv2.putText(
        frame,
        f"CAM 01  FRAME {frame_no + 1}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(frame)

writer.release()

print("\n--- Simulated CCTV Video ---")
print(f"Video created : {video_name}")
print(f"Frames       : {total_frames}")
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps}")

# ---------------------------------------------------------
# Step 2: Object-region detection
# ---------------------------------------------------------

cap = cv2.VideoCapture(video_name)

background = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=False
)

out = cv2.VideoWriter(
    output_name,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0
frames_with_detection = 0
max_objects = 0
saved_frame = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Background subtraction
    mask = background.apply(frame)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        kernel
    )

    # Find moving object regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_objects = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        detected_objects += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Object {detected_objects}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    if detected_objects > 0:
        frames_with_detection += 1

    max_objects = max(
        max_objects,
        detected_objects
    )

    cv2.putText(
        frame,
        f"Detected Objects: {detected_objects}",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    out.write(frame)

    # Save one representative annotated frame
    if detected_objects > 0 and not saved_frame:
        cv2.imwrite(frame_image, frame)
        saved_frame = True

cap.release()
out.release()

print("\n--- Detection Results ---")
print(f"Frames processed          : {frame_count}")
print(f"Frames with detection     : {frames_with_detection}")
print(f"Maximum objects detected  : {max_objects}")
print(f"Annotated video           : {output_name}")
print(f"Detection frame           : {frame_image}")

print("\n--- Execution Completed Successfully ---")