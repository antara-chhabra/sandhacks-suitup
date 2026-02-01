import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import threading
import time

# --- CONSTANTS & CONFIG ---
# MediaPipe Landmark Indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_EYE_CORNER_L = 33  # Inner corner right eye (viewer's left)
L_EYE_CORNER_R = 133  # Outer corner right eye
R_EYE_CORNER_L = 362  # Inner corner left eye
R_EYE_CORNER_R = 263  # Outer corner left eye
NOSE_TIP = 1


class BodyLanguageBot:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Critical for Iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State Variables
        self.emotion_result = "Neutral"
        self.gaze_status = "Centered"
        self.head_status = "Stable"
        self.nose_history = []
        self.frame_count = 0
        self.lock = threading.Lock()

    def get_gaze_ratio(self, eye_points, iris_center):
        """
        Calculates how centered the iris is within the eye.
        Returns a value between 0.0 (Left) and 1.0 (Right). 0.5 is Center.
        """
        # Distance between eye corners (Eye Width)
        eye_width = np.linalg.norm(eye_points[0] - eye_points[1])
        if eye_width == 0: return 0.5

        # Distance from Left Corner to Iris Center
        dist_to_center = np.linalg.norm(eye_points[0] - iris_center)

        # Ratio
        ratio = dist_to_center / eye_width
        return ratio

    def analyze_emotion_async(self, frame):
        """Runs DeepFace in a separate thread to avoid lag"""
        try:
            # Analyze takes ~200ms, so we offload it
            objs = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            with self.lock:
                self.emotion_result = objs[0]['dominant_emotion']
        except Exception as e:
            pass  # Ignore frames where face isn't clear

    def analyze_head_movement(self, nose_landmark, w, h):
        """Detects Nodding vs Shaking vs Stiffness"""
        x, y = nose_landmark.x * w, nose_landmark.y * h
        self.nose_history.append((x, y))

        if len(self.nose_history) > 30:  # Keep last ~1 second
            self.nose_history.pop(0)

        # Calculate Variance
        history = np.array(self.nose_history)
        std_x = np.std(history[:, 0])
        std_y = np.std(history[:, 1])

        # Thresholds (tuned for webcam distance)
        if std_y > 3.5 and std_y > std_x:
            return "Nodding (Good)"
        elif std_x > 3.5 and std_x > std_y:
            return "Shaking (Distracted?)"
        elif std_x < 1.0 and std_y < 1.0:
            return "Stiff (Relax!)"
        else:
            return "Natural"

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Flip for mirror effect & Convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            # 1. EMOTION ANALYSIS (Every 30 frames / 1 sec)
            if self.frame_count % 30 == 0:
                threading.Thread(target=self.analyze_emotion_async, args=(frame.copy(),)).start()

            # 2. FACE MESH PROCESSING
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                        results.multi_face_landmarks[0].landmark])

                # --- EYE GAZE LOGIC ---
                # Get coordinates
                left_eye_corners = mesh_points[[33, 133]]
                right_eye_corners = mesh_points[[362, 263]]
                left_iris = mesh_points[468]
                right_iris = mesh_points[473]

                # Draw Visuals (Eyes)
                cv2.circle(frame, left_iris, 2, (0, 255, 0), -1)
                cv2.circle(frame, right_iris, 2, (0, 255, 0), -1)

                # Calculate Ratios
                l_ratio = self.get_gaze_ratio(left_eye_corners, left_iris)
                r_ratio = self.get_gaze_ratio(right_eye_corners, right_iris)
                avg_ratio = (l_ratio + r_ratio) / 2

                if 0.42 < avg_ratio < 0.58:
                    self.gaze_status = "Eye Contact: GOOD"
                    color_gaze = (0, 255, 0)  # Green
                else:
                    self.gaze_status = "Eye Contact: LOOKING AWAY"
                    color_gaze = (0, 0, 255)  # Red

                # --- HEAD MOVEMENT LOGIC ---
                self.head_status = self.analyze_head_movement(results.multi_face_landmarks[0].landmark[NOSE_TIP], img_w,
                                                              img_h)

            else:
                self.gaze_status = "Face Not Found"
                color_gaze = (0, 0, 255)

            # 3. DISPLAY METRICS ON SCREEN
            # Box background
            cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 160), (255, 255, 255), 2)

            # Text
            cv2.putText(frame, self.gaze_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_gaze, 2)
            cv2.putText(frame, f"Emotion: {self.emotion_result}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Head: {self.head_status}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)

            cv2.imshow('Interview Coach - Body Language', frame)
            self.frame_count += 1

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    bot = BodyLanguageBot()
    bot.run()