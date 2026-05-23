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
        self.joy_pub = self.create_publisher(Float64MultiArray, '/hand_joystick', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper_cmd', 10)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.calibrated = False
        self.home_xyz = np.zeros(3)
        self.last_stable_z = 0.0 # Prevents the robot from pulling back when you grab
        
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        current_xyz = np.zeros(3)
        is_fist = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            
            # 1. Calculate robust Bounding Box Size for Depth
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            hand_size = np.sqrt((max(x_coords) - min(x_coords))**2 + (max(y_coords) - min(y_coords))**2)
            
            # 2. Fist Detection
            wrist_pos = np.array([landmarks[0].x, landmarks[0].y])
            finger_tips = [4, 8, 12, 16, 20]
            distances = [np.linalg.norm(np.array([landmarks[tip].x, landmarks[tip].y]) - wrist_pos) for tip in finger_tips]
            is_fist = bool(np.mean(distances) < 0.22)
            
            # 3. The Z-Axis Freeze (Stops the jerk-back bug)
            if is_fist:
                current_z = self.last_stable_z
            else:
                current_z = hand_size
                self.last_stable_z = hand_size
                
            current_xyz = np.array([landmarks[0].x, -landmarks[0].y, current_z])
            
            pinch_msg = Bool()
            pinch_msg.data = is_fist
            self.gripper_pub.publish(pinch_msg)
            
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        current_time = time.time()
        elapsed = current_time - self.start_time
        key = cv2.waitKey(1) & 0xFF

        if not self.calibrated:
            if elapsed < 7.0:
                cv2.putText(frame, f"HOLD HAND STILL: {7 - int(elapsed)}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if results.multi_hand_landmarks:
                    self.home_xyz = current_xyz
            else:
                self.calibrated = True
                self.get_logger().info("CALIBRATED!")
        elif key == ord('c') and results.multi_hand_landmarks:
            self.home_xyz = current_xyz

        if self.calibrated and results.multi_hand_landmarks:
            deltas = current_xyz - self.home_xyz
            msg = Float64MultiArray()
            msg.data = deltas.tolist()
            self.joy_pub.publish(msg)
            
            cv2.putText(frame, "JOYSTICK ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if is_fist:
                cv2.putText(frame, "GRAB ENGAGED", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Joystick Feed', frame)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
