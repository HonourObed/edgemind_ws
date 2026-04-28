import cv2
import mediapipe as mp
import numpy as np

# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def calculate_2d_angle(a, b, c):
    """
    Calculates the 2D angle between three pixel coordinates using atan2.
    a = shoulder, b = elbow (vertex), c = wrist
    """
    # math.atan2 returns the angle in radians from the X axis
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure we always get the acute/obtuse interior angle (<= 180)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

cap = cv2.VideoCapture(0)
print("Starting Camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.pose_landmarks:
        # Draw the body skeleton
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
       # Extract the Right Arm landmarks
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Get actual pixel dimensions to fix aspect ratio distortion
        h, w, _ = frame.shape
        
        # Convert to flat 2D pixel coordinates [x, y]
        shoulder_px = [shoulder.x * w, shoulder.y * h]
        elbow_px = [elbow.x * w, elbow.y * h]
        wrist_px = [wrist.x * w, wrist.y * h]
        
        # Calculate the true 2D interior angle
        elbow_angle = calculate_2d_angle(shoulder_px, elbow_px, wrist_px)
        
        # Display the data
        cv2.putText(frame, f"True Elbow Angle: {elbow_angle:.1f} deg", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw emphasis on the tracked arm vertex
        cv2.circle(frame, (int(elbow_px[0]), int(elbow_px[1])), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow('Arm Kinematics Extraction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()