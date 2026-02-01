"""
Body Language Detection System for Interview Preparation
Analyzes facial expressions, emotions, head movements, and eye contact
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
import time
from datetime import datetime
import os

# Try to import tensorflow/keras, provide installation instructions if missing
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not found. Install with: pip install tensorflow")
    exit(1)


class EmotionRecognizer:
    """Handles emotion recognition from facial images"""

    def __init__(self, model_path=None):
        self.emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy',
                               'Neutral', 'Sad', 'Surprise']
        self.model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. You'll need to train the model first.")

    def create_model(self):
        """Create CNN model for emotion recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotion_labels), activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_model(self, train_dir, test_dir, epochs=50, batch_size=64):
        """
        Train the emotion recognition model

        Args:
            train_dir: Path to training data directory
            test_dir: Path to testing data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("Creating model...")
        self.model = self.create_model()

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        print("Loading training data...")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical'
        )

        print("Loading validation data...")
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical'
        )

        print(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            verbose=1
        )

        # Save the model
        model_path = 'emotion_model.h5'
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        return history

    def load_model(self, model_path):
        """Load pre-trained model"""
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict_emotion(self, face_img):
        """
        Predict emotion from face image

        Args:
            face_img: Grayscale face image

        Returns:
            tuple: (emotion_label, confidence)
        """
        if self.model is None:
            return "Unknown", 0.0

        # Preprocess image
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        # Predict
        predictions = self.model.predict(face_img, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]

        return self.emotion_labels[emotion_idx], float(confidence)


class HeadPoseEstimator:
    """Estimates head pose and detects nodding"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Store head positions for nodding detection
        self.head_positions = deque(maxlen=30)  # Store last 30 frames
        self.nod_threshold = 15  # Degrees of movement to count as nod
        self.last_nod_time = 0
        self.nod_cooldown = 1.0  # Seconds between nods

    def estimate_head_pose(self, frame, face_landmarks):
        """
        Estimate head pose angles (pitch, yaw, roll)

        Returns:
            tuple: (pitch, yaw, roll) in degrees
        """
        img_h, img_w = frame.shape[:2]

        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # 2D image points from landmarks
        landmarks = face_landmarks.landmark
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),  # Nose tip
            (landmarks[152].x * img_w, landmarks[152].y * img_h),  # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),  # Left eye left corner
            (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye right corner
            (landmarks[61].x * img_w, landmarks[61].y * img_h),  # Left mouth corner
            (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth corner
        ], dtype="double")

        # Camera internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return pitch, yaw, roll

    def detect_nodding(self, pitch):
        """
        Detect nodding motion based on pitch changes

        Args:
            pitch: Current head pitch angle

        Returns:
            bool: True if nodding detected
        """
        self.head_positions.append(pitch)

        if len(self.head_positions) < 20:
            return False

        # Check for significant up-down movement
        recent_positions = list(self.head_positions)[-20:]
        max_pitch = max(recent_positions)
        min_pitch = min(recent_positions)
        pitch_range = max_pitch - min_pitch

        current_time = time.time()

        # Detect nod if significant movement and cooldown passed
        if pitch_range > self.nod_threshold and (current_time - self.last_nod_time) > self.nod_cooldown:
            self.last_nod_time = current_time
            return True

        return False


class EyeContactDetector:
    """Detects eye contact based on gaze direction"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.eye_contact_threshold = 25  # Degrees from center
        self.eye_contact_history = deque(maxlen=30)

    def detect_eye_contact(self, yaw, pitch):
        """
        Determine if user is making eye contact

        Args:
            yaw: Horizontal head angle
            pitch: Vertical head angle

        Returns:
            bool: True if making eye contact
        """
        # Eye contact when looking relatively straight at camera
        is_contact = abs(yaw) < self.eye_contact_threshold and abs(pitch) < self.eye_contact_threshold

        self.eye_contact_history.append(is_contact)

        return is_contact

    def get_eye_contact_percentage(self):
        """Get percentage of time maintaining eye contact"""
        if len(self.eye_contact_history) == 0:
            return 0.0

        return (sum(self.eye_contact_history) / len(self.eye_contact_history)) * 100


class InterviewAnalyzer:
    """Main class for analyzing interview body language"""

    def __init__(self, model_path=None):
        # Initialize components
        self.emotion_recognizer = EmotionRecognizer(model_path)
        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_contact_detector = EyeContactDetector()

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Tracking metrics
        self.metrics = {
            'emotions': [],
            'eye_contact_frames': 0,
            'total_frames': 0,
            'nods_detected': 0,
            'positive_emotions': 0,
            'negative_emotions': 0,
            'neutral_emotions': 0,
            'head_movements': []
        }

        self.start_time = None
        self.recording = False

    def analyze_frame(self, frame):
        """
        Analyze a single frame for body language cues

        Args:
            frame: BGR image from camera

        Returns:
            tuple: (annotated_frame, analysis_data)
        """
        if not self.recording:
            return frame, None

        self.metrics['total_frames'] += 1

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get face mesh results
        face_mesh_results = self.face_mesh.process(rgb_frame)

        analysis_data = {
            'emotion': 'Unknown',
            'confidence': 0.0,
            'eye_contact': False,
            'head_pose': (0, 0, 0),
            'nodding': False
        }

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face for emotion recognition
            face_roi = gray_frame[y:y + h, x:x + w]

            # Predict emotion
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_roi)
            analysis_data['emotion'] = emotion
            analysis_data['confidence'] = confidence

            # Track emotions
            self.metrics['emotions'].append(emotion)

            # Categorize emotions
            positive_emotions = ['Happy', 'Surprise']
            negative_emotions = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Sad']

            if emotion in positive_emotions:
                self.metrics['positive_emotions'] += 1
            elif emotion in negative_emotions:
                self.metrics['negative_emotions'] += 1
            else:
                self.metrics['neutral_emotions'] += 1

            # Display emotion
            emotion_text = f"{emotion} ({confidence * 100:.1f}%)"
            cv2.putText(frame, emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Head pose and eye contact analysis
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]

            # Estimate head pose
            pitch, yaw, roll = self.head_pose_estimator.estimate_head_pose(frame, face_landmarks)
            analysis_data['head_pose'] = (pitch, yaw, roll)

            # Store head movement
            self.metrics['head_movements'].append({
                'timestamp': time.time() - self.start_time if self.start_time else 0,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            })

            # Detect nodding
            is_nodding = self.head_pose_estimator.detect_nodding(pitch)
            if is_nodding:
                self.metrics['nods_detected'] += 1
                analysis_data['nodding'] = True

            # Detect eye contact
            eye_contact = self.eye_contact_detector.detect_eye_contact(yaw, pitch)
            if eye_contact:
                self.metrics['eye_contact_frames'] += 1
                analysis_data['eye_contact'] = True

            # Display head pose info
            pose_text = f"Pitch: {pitch:.1f} Yaw: {yaw:.1f} Roll: {roll:.1f}"
            cv2.putText(frame, pose_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display eye contact status
            eye_contact_text = "Eye Contact: YES" if eye_contact else "Eye Contact: NO"
            eye_contact_color = (0, 255, 0) if eye_contact else (0, 0, 255)
            cv2.putText(frame, eye_contact_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_contact_color, 2)

            # Display nodding
            if is_nodding:
                cv2.putText(frame, "NODDING DETECTED", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, analysis_data

    def start_recording(self):
        """Start recording metrics"""
        self.recording = True
        self.start_time = time.time()
        print("Recording started...")

    def stop_recording(self):
        """Stop recording and generate report"""
        self.recording = False
        print("Recording stopped. Generating report...")
        return self.generate_report()

    def generate_report(self):
        """
        Generate comprehensive analysis report

        Returns:
            dict: Analysis report with recommendations
        """
        if self.metrics['total_frames'] == 0:
            return {"error": "No data recorded"}

        # Calculate percentages
        eye_contact_pct = (self.metrics['eye_contact_frames'] / self.metrics['total_frames']) * 100

        # Emotion distribution
        from collections import Counter
        emotion_counts = Counter(self.metrics['emotions'])
        most_common_emotion = emotion_counts.most_common(1)[0] if emotion_counts else ("Unknown", 0)

        # Calculate emotion ratios
        total_emotions = len(self.metrics['emotions'])
        positive_pct = (self.metrics['positive_emotions'] / total_emotions * 100) if total_emotions > 0 else 0
        negative_pct = (self.metrics['negative_emotions'] / total_emotions * 100) if total_emotions > 0 else 0
        neutral_pct = (self.metrics['neutral_emotions'] / total_emotions * 100) if total_emotions > 0 else 0

        # Generate scores
        eye_contact_score = min(100, eye_contact_pct * 1.2)  # Scale up slightly
        emotion_score = positive_pct + (neutral_pct * 0.5) - (negative_pct * 0.3)
        engagement_score = min(100, (self.metrics['nods_detected'] / (self.metrics['total_frames'] / 30)) * 100)

        overall_score = (eye_contact_score * 0.4 + emotion_score * 0.4 + engagement_score * 0.2)

        report = {
            'overall_score': round(overall_score, 2),
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'total_frames_analyzed': self.metrics['total_frames'],

            'eye_contact': {
                'percentage': round(eye_contact_pct, 2),
                'score': round(eye_contact_score, 2),
                'frames_with_contact': self.metrics['eye_contact_frames']
            },

            'emotions': {
                'most_common': most_common_emotion[0],
                'distribution': dict(emotion_counts),
                'positive_percentage': round(positive_pct, 2),
                'negative_percentage': round(negative_pct, 2),
                'neutral_percentage': round(neutral_pct, 2),
                'score': round(emotion_score, 2)
            },

            'engagement': {
                'nods_detected': self.metrics['nods_detected'],
                'score': round(engagement_score, 2)
            },

            'recommendations': self.generate_recommendations(
                eye_contact_pct, positive_pct, negative_pct,
                self.metrics['nods_detected'], most_common_emotion[0]
            )
        }

        return report

    def generate_recommendations(self, eye_contact_pct, positive_pct,
                                 negative_pct, nods, dominant_emotion):
        """Generate personalized recommendations"""
        recommendations = []

        # Eye contact recommendations
        if eye_contact_pct < 40:
            recommendations.append({
                'category': 'Eye Contact',
                'severity': 'high',
                'message': 'Your eye contact is below optimal levels. Practice looking at the camera lens directly for 3-5 seconds at a time.',
                'tip': 'Imagine you are looking at the interviewer\'s eyes, not just the camera.'
            })
        elif eye_contact_pct < 60:
            recommendations.append({
                'category': 'Eye Contact',
                'severity': 'medium',
                'message': 'Good effort on eye contact, but there\'s room for improvement.',
                'tip': 'Try to maintain eye contact for 60-70% of the conversation.'
            })
        else:
            recommendations.append({
                'category': 'Eye Contact',
                'severity': 'low',
                'message': 'Excellent eye contact! Keep it up.',
                'tip': 'Continue this natural eye contact in real interviews.'
            })

        # Emotion recommendations
        if negative_pct > 30:
            recommendations.append({
                'category': 'Facial Expression',
                'severity': 'high',
                'message': f'You displayed negative emotions {negative_pct:.1f}% of the time.',
                'tip': 'Practice relaxation techniques before interviews. Take deep breaths and smile naturally.'
            })

        if positive_pct < 20:
            recommendations.append({
                'category': 'Facial Expression',
                'severity': 'medium',
                'message': 'Try to show more enthusiasm and positivity.',
                'tip': 'A genuine smile when appropriate can make a big difference.'
            })

        if dominant_emotion in ['Fear', 'Sad']:
            recommendations.append({
                'category': 'Emotion Control',
                'severity': 'high',
                'message': f'Your dominant emotion was {dominant_emotion}, which may indicate nervousness.',
                'tip': 'Practice mock interviews regularly to build confidence. Remember, the interviewer wants you to succeed!'
            })

        # Engagement recommendations
        if nods < 2:
            recommendations.append({
                'category': 'Engagement',
                'severity': 'medium',
                'message': 'Very few head nods detected. Nodding shows active listening.',
                'tip': 'Nod occasionally while the interviewer speaks to show engagement and understanding.'
            })
        elif nods > 20:
            recommendations.append({
                'category': 'Engagement',
                'severity': 'low',
                'message': 'You nod frequently - ensure it appears natural and not excessive.',
                'tip': 'Nod at key moments, not continuously.'
            })

        # Overall encouragement
        if len(recommendations) == 0 or all(r['severity'] == 'low' for r in recommendations):
            recommendations.append({
                'category': 'Overall',
                'severity': 'low',
                'message': 'Great performance! You demonstrate strong body language skills.',
                'tip': 'Keep practicing to maintain these excellent habits.'
            })

        return recommendations


def run_interview_practice(model_path=None, duration=60):
    """
    Run an interactive interview practice session

    Args:
        model_path: Path to trained emotion model (optional)
        duration: Maximum duration in seconds (0 = unlimited)
    """
    print("\n" + "=" * 60)
    print("INTERVIEW BODY LANGUAGE ANALYZER")
    print("=" * 60)
    print("\nThis tool will analyze your:")
    print("  • Facial expressions and emotions")
    print("  • Eye contact")
    print("  • Head movements (nodding)")
    print("  • Overall body language\n")

    analyzer = InterviewAnalyzer(model_path)

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    print("Camera opened successfully!")
    print("\nControls:")
    print("  Press 'S' to START recording")
    print("  Press 'Q' to STOP and generate report")
    print("  Press 'ESC' to exit without report")

    recording_started = False
    start_time = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Analyze frame
        annotated_frame, _ = analyzer.analyze_frame(frame)

        # Add recording indicator
        if analyzer.recording:
            elapsed = time.time() - start_time
            cv2.circle(annotated_frame, (30, 120), 10, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"REC {elapsed:.1f}s", (50, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if duration > 0 and elapsed >= duration:
                print("\nTime limit reached!")
                break
        else:
            cv2.putText(annotated_frame, "Press 'S' to start", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Interview Practice - Body Language Analysis', annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            if not analyzer.recording:
                analyzer.start_recording()
                start_time = time.time()
                recording_started = True

        elif key == ord('q') or key == ord('Q'):
            if analyzer.recording:
                break
            else:
                print("\nStart recording first before generating report!")

        elif key == 27:  # ESC
            print("\nExiting without report...")
            cap.release()
            cv2.destroyAllWindows()
            return

    # Stop recording and generate report
    if recording_started:
        report = analyzer.stop_recording()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Display report
        display_report(report)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"interview_report_{timestamp}.txt"
        save_report_to_file(report, report_file)
        print(f"\nReport saved to: {report_file}")
    else:
        cap.release()
        cv2.destroyAllWindows()


def display_report(report):
    """Display the analysis report in console"""
    print("\n" + "=" * 60)
    print("INTERVIEW PERFORMANCE REPORT")
    print("=" * 60)

    print(f"\nOverall Score: {report['overall_score']:.1f}/100")
    print(f"Duration: {report['duration_seconds']:.1f} seconds")
    print(f"Frames Analyzed: {report['total_frames_analyzed']}")

    print("\n" + "-" * 60)
    print("EYE CONTACT ANALYSIS")
    print("-" * 60)
    print(f"Percentage: {report['eye_contact']['percentage']:.1f}%")
    print(f"Score: {report['eye_contact']['score']:.1f}/100")
    print(f"Frames with Contact: {report['eye_contact']['frames_with_contact']}")

    print("\n" + "-" * 60)
    print("EMOTION ANALYSIS")
    print("-" * 60)
    print(f"Most Common Emotion: {report['emotions']['most_common']}")
    print(f"Positive Emotions: {report['emotions']['positive_percentage']:.1f}%")
    print(f"Negative Emotions: {report['emotions']['negative_percentage']:.1f}%")
    print(f"Neutral Emotions: {report['emotions']['neutral_percentage']:.1f}%")
    print(f"Emotion Score: {report['emotions']['score']:.1f}/100")

    print("\nEmotion Distribution:")
    for emotion, count in sorted(report['emotions']['distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count}")

    print("\n" + "-" * 60)
    print("ENGAGEMENT ANALYSIS")
    print("-" * 60)
    print(f"Nods Detected: {report['engagement']['nods_detected']}")
    print(f"Engagement Score: {report['engagement']['score']:.1f}/100")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    for i, rec in enumerate(report['recommendations'], 1):
        severity_symbol = {
            'high': '⚠️  [HIGH PRIORITY]',
            'medium': '⚡ [MEDIUM]',
            'low': '✓  [STRENGTH]'
        }

        print(f"\n{i}. {severity_symbol[rec['severity']]} {rec['category']}")
        print(f"   {rec['message']}")
        print(f"   💡 Tip: {rec['tip']}")

    print("\n" + "=" * 60)


def save_report_to_file(report, filename):
    """Save report to text file"""
    with open(filename, 'w') as f:
        f.write("INTERVIEW PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Overall Score: {report['overall_score']:.1f}/100\n")
        f.write(f"Duration: {report['duration_seconds']:.1f} seconds\n")
        f.write(f"Frames Analyzed: {report['total_frames_analyzed']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("EYE CONTACT ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Percentage: {report['eye_contact']['percentage']:.1f}%\n")
        f.write(f"Score: {report['eye_contact']['score']:.1f}/100\n\n")

        f.write("-" * 60 + "\n")
        f.write("EMOTION ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Most Common: {report['emotions']['most_common']}\n")
        f.write(f"Positive: {report['emotions']['positive_percentage']:.1f}%\n")
        f.write(f"Negative: {report['emotions']['negative_percentage']:.1f}%\n")
        f.write(f"Neutral: {report['emotions']['neutral_percentage']:.1f}%\n\n")

        f.write("-" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 60 + "\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"\n{i}. [{rec['severity'].upper()}] {rec['category']}\n")
            f.write(f"   {rec['message']}\n")
            f.write(f"   Tip: {rec['tip']}\n")


def train_emotion_model(train_dir, test_dir):
    """
    Train the emotion recognition model

    Args:
        train_dir: Path to training data directory (should contain subdirectories for each emotion)
        test_dir: Path to testing data directory
    """
    print("\n" + "=" * 60)
    print("EMOTION MODEL TRAINING")
    print("=" * 60)

    recognizer = EmotionRecognizer()

    print("\nStarting training process...")
    print("This may take a while depending on your hardware...\n")

    history = recognizer.train_model(train_dir, test_dir, epochs=50)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("Model saved as 'emotion_model.h5'")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("INTERVIEW BODY LANGUAGE ANALYZER")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Run interview practice (requires trained model)")
    print("2. Train emotion recognition model")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        model_path = input("\nEnter path to trained model (or press Enter for 'emotion_model.h5'): ").strip()
        if not model_path:
            model_path = "emotion_model.h5"

        if not os.path.exists(model_path):
            print(f"\nModel not found at {model_path}")
            print("Please train the model first (option 2) or provide correct path.")
        else:
            duration = input("\nEnter duration in seconds (0 for unlimited): ").strip()
            duration = int(duration) if duration.isdigit() else 60

            run_interview_practice(model_path, duration)

    elif choice == "2":
        print("\nTo train the model, you need the Kaggle dataset:")
        print("https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data")
        print("\nExtract it and provide paths to 'train' and 'test' directories.")

        train_dir = input("\nEnter path to training directory: ").strip()
        test_dir = input("Enter path to testing directory: ").strip()

        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_emotion_model(train_dir, test_dir)
        else:
            print("\nError: Invalid directory paths!")

    elif choice == "3":
        print("\nGoodbye!")

    else:
        print("\nInvalid choice!")