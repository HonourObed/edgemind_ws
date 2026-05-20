import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
import cv2
import mediapipe as mp
import numpy as np
import time

class UnifiedVisionNode(Node):
    def __init__(self):
        super().__init__('unified_vision')
        self.angle_pub = self.create_publisher(Float64MultiArray, '/arm_angles', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper_cmd', 10)
        
        self.mp_pose = mp.solutions.pose
        # Using model_complexity=1 for better 3D depth extraction
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        
        # --- CALIBRATION SYSTEM ---
        self.start_time = time.time()
        self.calibrated = False
        self.baseline_angles = np.zeros(5) # [Sh_Yaw, Sh_Pitch, El_Pitch, Wr_Pitch, Wr_Yaw]
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("3D Vision Node Online: Standby for 7-Second Calibration...")

    def get_3d_angles(self, world_landmarks):
        # Extract 3D points (x: left/right, y: up/down, z: depth)
        rs = np.array([world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z])
        re = np.array([world_landmarks[14].x, world_landmarks[14].y, world_landmarks[14].z])
        rw = np.array([world_landmarks[16].x, world_landmarks[16].y, world_landmarks[16].z])
        
        # Calculate Vectors
        upper_arm = re - rs
        forearm = rw - re
        
        # Calculate Spherical Angles (Radians)
        sh_yaw = np.arctan2(upper_arm[2], upper_arm[0])
        sh_pitch = np.arctan2(upper_arm[1], np.sqrt(upper_arm[0]**2 + upper_arm[2]**2))
        
        # Elbow Pitch: Angle between upper arm and forearm vectors
        cos_theta = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm) + 1e-6)
        el_pitch = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        wr_yaw = np.arctan2(forearm[2], forearm[0])
        wr_pitch = np.arctan2(forearm[1], np.sqrt(forearm[0]**2 + forearm[2]**2))
        
        return np.array([sh_yaw, sh_pitch, el_pitch, wr_pitch, wr_yaw])

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        current_angles = np.zeros(5)
        spiderman_gesture = False
        
        if pose_results.pose_world_landmarks:
            current_angles = self.get_3d_angles(pose_results.pose_world_landmarks.landmark)
            
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0].landmark
            
            # 1. Gripper Logic (Thumb to Index Distance)
            thumb_tip = np.array([hand[4].x, hand[4].y])
            index_tip = np.array([hand[8].x, hand[8].y])
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            pinch_msg = Bool()
            pinch_msg.data = bool(pinch_dist < 0.05)
            self.gripper_pub.publish(pinch_msg)
            
            # 2. Spiderman Gesture Detection (Thwip)
            # Index & Pinky Extended (Tip higher than PIP joint)
            idx_up = hand[8].y < hand[6].y
            pnk_up = hand[20].y < hand[18].y
            # Middle & Ring Folded (Tip lower than PIP joint)
            mid_dn = hand[12].y > hand[10].y
            rng_dn = hand[16].y > hand[14].y
            
            if idx_up and pnk_up and mid_dn and rng_dn:
                spiderman_gesture = True

        # --- THE STATE MACHINE ---
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        key = cv2.waitKey(1) & 0xFF
        trigger_manual_cal = (key == ord('c') or spiderman_gesture)

        if not self.calibrated:
            if elapsed < 7.0:
                cv2.putText(frame, f"CALIBRATING IN: {7 - int(elapsed)}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, "STAND IN T-POSE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.baseline_angles = current_angles
                self.calibrated = True
                self.get_logger().info("BASELINES LOCKED! You have control.")
        elif trigger_manual_cal:
            self.baseline_angles = current_angles
            self.get_logger().info("MANUAL OVERRIDE: Recalibrated Home State.")
            cv2.putText(frame, "RECALIBRATED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Calculate Delta and Publish (Only if Calibrated)
        if self.calibrated and pose_results.pose_world_landmarks:
            delta_angles = current_angles - self.baseline_angles
            msg = Float64MultiArray()
            msg.data = delta_angles.tolist()
            self.angle_pub.publish(msg)
            
            cv2.putText(frame, "SYSTEM ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw UI
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            
        cv2.imshow('Teleoperation Feed', frame)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
