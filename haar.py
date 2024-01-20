import streamlit as st
import cv2
import numpy as np
import time

# Gunakan path lengkap untuk haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = None
is_camera_running = False

blink_threshold = 5  # Adjust this threshold for blink sensitivity
blink_counter = 0
sleep_counter = 0
eyes_closed_start_time = None
sound_duration = 0.5  # Duration in seconds to play sound for prolonged eye closure

def detect_tiredness(frame):
    global blink_counter, sleep_counter, eyes_closed_start_time

    if frame is None:
        st.warning("Failed to capture frame.")
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    frame_copy = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame_copy[y:y+h, x:x+w]

        # Assuming the eyes are in the face region (you may need to fine-tune this)
        left_blink = blinked(roi_gray, landmarks=None)  # Pass None for landmarks
        right_blink = left_blink  # For simplicity, assuming both eyes blink similarly

        status = None
        color = None

        # Detect prolonged eye closure
        if left_blink == 0 or right_blink == 0:
            blink_counter += 1
            sleep_counter += 1

            if blink_counter > blink_threshold:
                if eyes_closed_start_time is None:
                    # Start the timer when eyes are closed
                    eyes_closed_start_time = time.time()
                else:
                    # Check if the eyes have been closed for the specified duration
                    elapsed_time = time.time() - eyes_closed_start_time
                    if elapsed_time > sound_duration:
                        status = "SLEEP !!!"
                        color = (255, 0, 0)

                        # You can add code here to play sound for prolonged eye closure

        # Reset counters and timer if eyes are open
        elif left_blink == 1 or right_blink == 1:
            blink_counter = 0
            sleep_counter = 0
            eyes_closed_start_time = None

        else:
            status = "Active :)"
            color = (0, 255, 0)

        if status:
            cv2.putText(frame_copy, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return frame_copy

def start_camera():
    global cap, is_camera_running

    if not is_camera_running:
        cap = cv2.VideoCapture(0)
        st.success("Camera started successfully!")
        is_camera_running = True

def stop_camera():
    global cap, is_camera_running

    if is_camera_running:
        cap.release()
        is_camera_running = False

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(roi_gray, landmarks=None):
    # Implement your logic to detect blinking in the region of interest (roi_gray)
    # You may use additional libraries or algorithms for this task
    # Return 2 for closed eyes, 1 for partial blink, 0 for open eyes
    # Example: You can use the aspect ratio of eye landmarks for blink detection

    return 0

def main():
    st.title("Tired Detection App")

    # Button to start the camera
    if not is_camera_running:
        if st.button("Start Camera"):
            start_camera()

    # Button to stop the camera
    if is_camera_running:
        if st.button("Stop Camera"):
            stop_camera()

    st.markdown("---")

    # Video Stream
    if is_camera_running:
        video_stream = st.empty()
        while True:
            _, frame = cap.read()

            # Ensure the frame is not None
            if frame is not None:
                frame_copy = detect_tiredness(frame)

                # Display the frame with text overlay
                video_stream.image(frame_copy, channels="BGR", caption="Video Stream", use_column_width=True)

if __name__ == '__main__':
    main()
