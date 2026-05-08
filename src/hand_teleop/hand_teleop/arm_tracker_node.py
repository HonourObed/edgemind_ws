import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import cv2
import mediapipe as mp
import numpy as np

class ArmTrackerNode(Node):
    def __init__(self):
        super().__init__('arm_tracker')
        
        # Broadcasts a standard decimal number on '/elbow_angle'
        self.publisher_ = self.create_publisher(Float64, '/elbow_angle', 10)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        
        # Run at 20 Hz (20 frames per second)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('Arm Tracker broadcasting on /elbow_angle...')

    def calculate_2d_angle(self, a, b, c):
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360.0 - angle
        return angle

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            h, w, _ = frame.shape
            shoulder_px = [shoulder.x * w, shoulder.y * h]
            elbow_px = [elbow.x * w, elbow.y * h]
            wrist_px = [wrist.x * w, wrist.y * h]
            
            angle = self.calculate_2d_angle(shoulder_px, elbow_px, wrist_px)
            
            # Publish to ROS 2
            msg = Float64()
            msg.data = float(angle)
            self.publisher_.publish(msg)
            
            cv2.putText(frame, f"Publishing Angle: {angle:.1f}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ROS 2 Arm Publisher', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArmTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()