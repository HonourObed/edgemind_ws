import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
import cv2
import mediapipe as mp
import numpy as np
import threading
import math

class UnifiedVisionNode(Node):
    def __init__(self):
        super().__init__('unified_vision_node')
        
        # 1. Initialize Both Publishers
        self.angle_pub_ = self.create_publisher(Float64, '/elbow_angle', 10)
        self.gripper_pub_ = self.create_publisher(Bool, '/gripper_cmd', 10)
        
        # 2. Initialize MediaPipe Models
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.pinch_threshold = 0.05
        
        # 3. Open Camera ONCE
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            
        self.get_logger().info('Unified Vision Node Active: Tracking Arm and Hand simultaneously...')
        
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def calculate_2d_angle(self, a, b, c):
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def calculate_distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

    def _camera_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Process BOTH models on the single frame
            pose_results = self.pose.process(image_rgb)
            hand_results = self.hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            
            # --- POSE LOGIC (Elbow Angle) ---
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                landmarks = pose_results.pose_landmarks.landmark
                shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                if shoulder.visibility > 0.5 and elbow.visibility > 0.5 and wrist.visibility > 0.5:
                    h, w, _ = frame.shape
                    shoulder_px = [shoulder.x * w, shoulder.y * h]
                    elbow_px = [elbow.x * w, elbow.y * h]
                    wrist_px = [wrist.x * w, wrist.y * h]
                    
                    angle = self.calculate_2d_angle(shoulder_px, elbow_px, wrist_px)
                    
                    angle_msg = Float64()
                    angle_msg.data = float(angle)
                    self.angle_pub_.publish(angle_msg)
                    
                    cv2.putText(frame, f"Arm Angle: {angle:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- HAND LOGIC (Pinch Detector) ---
            gripper_msg = Bool()
            gripper_msg.data = False # Default Open
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    
                    distance = self.calculate_distance(thumb_tip, index_tip)
                    gripper_msg.data = bool(distance < self.pinch_threshold)
                    
                    state_text = "GRAB" if gripper_msg.data else "OPEN"
                    color = (0, 255, 0) if gripper_msg.data else (0, 0, 255)
                    cv2.putText(frame, f"Gripper: {state_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
            self.gripper_pub_.publish(gripper_msg)

            cv2.imshow('Unified Teleoperation Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
