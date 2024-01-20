import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time


class TiredDetectionApp:
    def __init__(self):
        self.cap = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.is_camera_running = False

        self.blink_threshold = 5  # Adjust this threshold for blink sensitivity
        self.blink_counter = 0
        self.sleep_counter = 0
        self.eyes_closed_start_time = None
        self.sound_duration = 0.5  # Duration in seconds to play sound for prolonged eye closure

    def detect_tiredness(self, frame):
        if frame is None:
            st.warning("Failed to capture frame.")
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        frame_copy = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = self.predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = self.blinked(landmarks[36], landmarks[37],
                                      landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = self.blinked(landmarks[42], landmarks[43],
                                       landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            status = None
            color = None

            # Detect prolonged eye closure
            if left_blink == 0 or right_blink == 0:
                self.blink_counter += 1
                self.sleep_counter += 1

                if self.blink_counter > self.blink_threshold:
                    if self.eyes_closed_start_time is None:
                        # Start the timer when eyes are closed
                        self.eyes_closed_start_time = time.time()
                    else:
                        # Check if the eyes have been closed for the specified duration
                        elapsed_time = time.time() - self.eyes_closed_start_time
                        if elapsed_time > self.sound_duration:
                            status = "SLEEP !!!"
                            color = (255, 0, 0)


            # Reset counters and timer if eyes are open
            elif left_blink == 1 or right_blink == 1:
                self.blink_counter = 0
                self.sleep_counter = 0
                self.eyes_closed_start_time = None

            else:
                status = "Active :)"
                color = (0, 255, 0)

            if status:
                cv2.putText(frame_copy, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(frame_copy, (x, y), 1, (255, 255, 255), -1)

        return frame_copy

    def start_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            st.success("Camera started successfully!")
            self.is_camera_running = True

    def stop_camera(self):
        if self.is_camera_running:
            self.cap.release()
            self.is_camera_running = False

    def run(self):
        st.title("Tired Detection App")

        # Button to start the camera
        if not self.is_camera_running:
            if st.button("Start Camera"):
                self.start_camera()

        # Button to stop the camera
        if self.is_camera_running:
            if st.button("Stop Camera"):
                self.stop_camera()

        st.markdown("---")

        # Video Stream
        if self.is_camera_running:
            video_stream = st.empty()
            while True:
                _, frame = self.cap.read()

                # Ensure the frame is not None
                if frame is not None:
                    frame_copy = self.detect_tiredness(frame)

                    # Display the frame with text overlay
                    video_stream.image(frame_copy, channels="BGR", caption="Video Stream", use_column_width=True)

    def on_stop(self):
        if self.cap is not None:
            self.cap.release()

    def compute(self, ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(self, a, b, c, d, e, f):
        up = self.compute(b, d) + self.compute(c, e)
        down = self.compute(a, f)
        ratio = up / (2.0 * down)

        if ratio > 0.25:
            return 2
        elif 0.21 < ratio <= 0.25:
            return 1
        else:
            return 0

def main():
    app = TiredDetectionApp()
    app.run()
    app.on_stop()

if __name__ == '__main__':
    main()
