import cv2
import time
import numpy as np
from eyepop import EyePopSdk
import os
from dotenv import load_dotenv
from eyepop import EyePopSdk 

load_dotenv()


# ==========================================
# 1. CONFIGURATION
# ==========================================
# PASTE YOUR KEYS HERE AGAIN


# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
class InterviewAnalyzer:
    def __init__(self):
        self.pitch_history = []
        self.last_wrist_pos = None
        self.history_len = 30 

    def check_eye_contact(self, face_mesh):
        if not face_mesh: return "No Face"
        # Keypoint 1 is typically the nose tip
        nose = face_mesh[1] 
        x, y = nose['x'], nose['y']
        # Confidence Box (Center 30% of screen)
        if 0.35 < x < 0.65 and 0.35 < y < 0.65:
            return "Good Eye Contact"
        return "Looking Away"

    def check_nodding(self, head_pose):
        if not head_pose: return "Unknown"
        self.pitch_history.append(head_pose.get('pitch', 0))
        if len(self.pitch_history) > self.history_len: 
            self.pitch_history.pop(0)
        variance = np.var(self.pitch_history)
        if variance > 0.05: return "Nodding / Active"
        return "Still"

    def check_fidget(self, hands):
        if not hands or len(hands) == 0: return "Hands Hidden"
        hand = hands[0]
        if 'points' in hand:
            wrist = hand['points'][0]
        else:
            return "No Hand Data"
        
        current_pos = np.array([wrist['x'], wrist['y']])
        if self.last_wrist_pos is None:
            self.last_wrist_pos = current_pos
            return "Calm"
        
        speed = np.linalg.norm(current_pos - self.last_wrist_pos)
        self.last_wrist_pos = current_pos
        
        if speed > 0.10: return "Gesturing"
        if speed > 0.02: return "Fidgeting"
        return "Calm"

# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_vision_layer():
    analyzer = InterviewAnalyzer()
    
    # Try Camera 0 first, then 1 if it fails
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera 0 failed, trying Camera 1...")
        cap = cv2.VideoCapture(1)

    print("Connecting to EyePop...")
    
    # --- THE FIX IS HERE: using 'workerEndpoint' ---
    
    api_key = os.getenv("EYEPOP_API_KEY")

    if not api_key:
        raise RuntimeError("EYEPOP_API_KEY not found in .env file")

    with EyePopSdk.workerEndpoint() as endpoint:
        print("Connected! Processing video... Press 'q' to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            try:
                results = endpoint.upload(frame).predict()
            except Exception as e:
                print(f"Connection glitch: {e}")
                continue
            
            eye_status = "Scanning..."
            nod_status = "Scanning..."
            fidget_status = "Scanning..."

            if results and 'objects' in results:
                for person in results['objects']:
                    if person.get('classLabel') == 'person':
                        face_mesh = person.get('faceMesh', {}).get('points', [])
                        head_pose = person.get('headPose', {})
                        hands = person.get('hands', [])

                        eye_status = analyzer.check_eye_contact(face_mesh)
                        nod_status = analyzer.check_nodding(head_pose)
                        fidget_status = analyzer.check_fidget(hands)
                        break 

            # Overlay Results
            color_eye = (0, 255, 0) if "Good" in eye_status else (0, 0, 255)
            cv2.putText(frame, f"Eye Contact: {eye_status}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_eye, 2)
            cv2.putText(frame, f"Head: {nod_status}", (30, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Hands: {fidget_status}", (30, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Interview Prep Bot', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_layer()