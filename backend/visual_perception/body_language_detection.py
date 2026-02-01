"""
Body Language Detection System for Interview Preparation
Analyzes facial expressions, emotions, head movements, and eye contact
NO MEDIAPIPE REQUIRED - Uses OpenCV only
"""

import cv2
import numpy as np
import pandas as pd
from collections import deque
import time
from datetime import datetime
import os

# Try to import tensorflow/keras
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not found. Install with: pip install tensorflow")
    exit(1)


class EmotionRecognizer:
    """Handles emotion recognition from facial images"""

    def __init__(self, model_path=None):
        # Updated to match actual dataset classes (7 emotions)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy',
                               'Neutral', 'Sad', 'Surprise']
        self.model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found at:", model_path)

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
    """Estimates head pose and detects nodding using OpenCV only"""

    def __init__(self):
        # Load face landmark detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # Store head positions for nodding detection
        self.head_positions = deque(maxlen=30)
        self.nod_threshold = 20  # Pixels of movement to count as nod
        self.last_nod_time = 0
        self.nod_cooldown = 1.0

        # Store previous face position
        self.prev_face_y = None

    def estimate_head_pose_simple(self, frame, face_rect):
        """
        Estimate head pose using simple geometric approach

        Args:
            frame: BGR image
            face_rect: (x, y, w, h) of detected face

        Returns:
            tuple: (pitch, yaw, roll) in degrees (approximations)
        """
        x, y, w, h = face_rect

        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)

        pitch = 0  # Up/down tilt
        yaw = 0    # Left/right turn
        roll = 0   # Head tilt

        if len(eyes) >= 2:
            # Sort eyes by x coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]

            # Calculate eye centers
            left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)

            # Calculate roll (head tilt) from eye alignment
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            roll = np.degrees(np.arctan2(dy, dx))

            # Estimate yaw from eye distance to face center
            face_center_x = w // 2
            eye_midpoint_x = (left_eye_center[0] + right_eye_center[0]) // 2
            yaw_offset = eye_midpoint_x - face_center_x
            yaw = (yaw_offset / (w / 2)) * 30  # Scale to degrees

            # Estimate pitch from eye position relative to face height
            eye_avg_y = (left_eye_center[1] + right_eye_center[1]) // 2
            face_center_y = h // 2
            pitch_offset = eye_avg_y - face_center_y
            pitch = (pitch_offset / (h / 2)) * 30  # Scale to degrees

        return pitch, yaw, roll

    def detect_nodding(self, face_y):
        """
        Detect nodding motion based on face Y position changes

        Args:
            face_y: Current Y position of face

        Returns:
            bool: True if nodding detected
        """
        self.head_positions.append(face_y)

        if len(self.head_positions) < 15:
            return False

        # Check for significant up-down movement
        recent_positions = list(self.head_positions)[-15:]
        max_y = max(recent_positions)
        min_y = min(recent_positions)
        y_range = max_y - min_y

        current_time = time.time()

        # Detect nod if significant movement and cooldown passed
        if y_range > self.nod_threshold and (current_time - self.last_nod_time) > self.nod_cooldown:
            self.last_nod_time = current_time
            return True

        return False


class EyeContactDetector:
    """Detects eye contact based on gaze direction"""

    def __init__(self):
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

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract face for emotion recognition
            face_roi = gray_frame[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_roi)
            analysis_data['emotion'] = emotion
            analysis_data['confidence'] = confidence

            # Track emotions
            self.metrics['emotions'].append(emotion)

            # Categorize emotions (7 classes)
            positive_emotions = ['Happy', 'Surprise']
            negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

            if emotion in positive_emotions:
                self.metrics['positive_emotions'] += 1
            elif emotion in negative_emotions:
                self.metrics['negative_emotions'] += 1
            else:
                self.metrics['neutral_emotions'] += 1

            # Display emotion
            emotion_text = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, emotion_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Estimate head pose
            pitch, yaw, roll = self.head_pose_estimator.estimate_head_pose_simple(frame, (x, y, w, h))
            analysis_data['head_pose'] = (pitch, yaw, roll)

            # Store head movement
            self.metrics['head_movements'].append({
                'timestamp': time.time() - self.start_time if self.start_time else 0,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            })

            # Detect nodding
            face_center_y = y + h // 2
            is_nodding = self.head_pose_estimator.detect_nodding(face_center_y)
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
        """Generate comprehensive analysis report"""
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
        eye_contact_score = min(100, eye_contact_pct * 1.2)
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


def display_report(report):
    """Display the analysis report in console"""
    print("\n" + "="*60)
    print("INTERVIEW PERFORMANCE REPORT")
    print("="*60)

    print(f"\nOverall Score: {report['overall_score']:.1f}/100")
    print(f"Duration: {report['duration_seconds']:.1f} seconds")
    print(f"Frames Analyzed: {report['total_frames_analyzed']}")

    print("\n" + "-"*60)
    print("EYE CONTACT ANALYSIS")
    print("-"*60)
    print(f"Percentage: {report['eye_contact']['percentage']:.1f}%")
    print(f"Score: {report['eye_contact']['score']:.1f}/100")
    print(f"Frames with Contact: {report['eye_contact']['frames_with_contact']}")

    print("\n" + "-"*60)
    print("EMOTION ANALYSIS")
    print("-"*60)
    print(f"Most Common Emotion: {report['emotions']['most_common']}")
    print(f"Positive Emotions: {report['emotions']['positive_percentage']:.1f}%")
    print(f"Negative Emotions: {report['emotions']['negative_percentage']:.1f}%")
    print(f"Neutral Emotions: {report['emotions']['neutral_percentage']:.1f}%")
    print(f"Emotion Score: {report['emotions']['score']:.1f}/100")

    print("\nEmotion Distribution:")
    for emotion, count in sorted(report['emotions']['distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count}")

    print("\n" + "-"*60)
    print("ENGAGEMENT ANALYSIS")
    print("-"*60)
    print(f"Nods Detected: {report['engagement']['nods_detected']}")
    print(f"Engagement Score: {report['engagement']['score']:.1f}/100")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    for i, rec in enumerate(report['recommendations'], 1):
        severity_symbol = {
            'high': '⚠️  [HIGH PRIORITY]',
            'medium': '⚡ [MEDIUM]',
            'low': '✓  [STRENGTH]'
        }

        print(f"\n{i}. {severity_symbol[rec['severity']]} {rec['category']}")
        print(f"   {rec['message']}")
        print(f"   💡 Tip: {rec['tip']}")

    print("\n" + "="*60)


def save_report_to_file(report, filename):
    """Save report to text file"""
    with open(filename, 'w') as f:
        f.write("INTERVIEW PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")

        f.write(f"Overall Score: {report['overall_score']:.1f}/100\n")
        f.write(f"Duration: {report['duration_seconds']:.1f} seconds\n")
        f.write(f"Frames Analyzed: {report['total_frames_analyzed']}\n\n")

        f.write("-"*60 + "\n")
        f.write("EYE CONTACT ANALYSIS\n")
        f.write("-"*60 + "\n")
        f.write(f"Percentage: {report['eye_contact']['percentage']:.1f}%\n")
        f.write(f"Score: {report['eye_contact']['score']:.1f}/100\n\n")

        f.write("-"*60 + "\n")
        f.write("EMOTION ANALYSIS\n")
        f.write("-"*60 + "\n")
        f.write(f"Most Common: {report['emotions']['most_common']}\n")
        f.write(f"Positive: {report['emotions']['positive_percentage']:.1f}%\n")
        f.write(f"Negative: {report['emotions']['negative_percentage']:.1f}%\n")
        f.write(f"Neutral: {report['emotions']['neutral_percentage']:.1f}%\n\n")

        f.write("-"*60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*60 + "\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"\n{i}. [{rec['severity'].upper()}] {rec['category']}\n")
            f.write(f"   {rec['message']}\n")
            f.write(f"   Tip: {rec['tip']}\n")


def run_interview_practice(model_path=None, duration=60):
    """Run an interactive interview practice session"""
    print("\n" + "="*60)
    print("INTERVIEW BODY LANGUAGE ANALYZER")
    print("="*60)
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


if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("INTERVIEW BODY LANGUAGE ANALYZER")
    print("(OpenCV-only version - No MediaPipe required)")
    print("="*60)

    # Check if emotion model exists
    model_path = "emotion_model.h5"

    # Look in parent directory if not found
    if not os.path.exists(model_path):
        parent_model = "../emotion_model.h5"
        if os.path.exists(parent_model):
            model_path = parent_model
        else:
            print(f"\nWARNING: Emotion model not found at {model_path}")
            print("The script will run but emotion detection will not work.")
            print("Make sure emotion_model.h5 is in the same directory as this script.\n")
            model_path = None

    duration = input("\nEnter duration in seconds (0 for unlimited, default=60): ").strip()
    duration = int(duration) if duration.isdigit() else 60

    run_interview_practice(model_path, duration)