import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import cv2
import mediapipe as mp
import numpy as np
import threading

class ArmTrackerNode(Node):
    def __init__(self):
        super().__init__('arm_tracker')

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
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")

        self.get_logger().info('Arm Tracker starting camera thread...')

        # Run camera loop in a separate thread so it never blocks rclpy.spin()
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def calculate_2d_angle(self, a, b, c):
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def _camera_loop(self):
        VISIBILITY_THRESHOLD = 0.5
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Camera read failed.', throttle_duration_sec=2.0)
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark
                shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow    = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist    = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

                if (shoulder.visibility < VISIBILITY_THRESHOLD or
                        elbow.visibility < VISIBILITY_THRESHOLD or
                        wrist.visibility < VISIBILITY_THRESHOLD):
                    cv2.putText(frame, "ARM NOT FULLY VISIBLE - Move back!", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    h, w, _ = frame.shape
                    shoulder_px = [shoulder.x * w, shoulder.y * h]
                    elbow_px    = [elbow.x * w,    elbow.y * h]
                    wrist_px    = [wrist.x * w,    wrist.y * h]

                    angle = self.calculate_2d_angle(shoulder_px, elbow_px, wrist_px)

                    msg = Float64()
                    msg.data = float(angle)
                    self.publisher_.publish(msg)

                    cv2.putText(frame, f"Publishing Angle: {angle:.1f}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"S:{shoulder.visibility:.2f} E:{elbow.visibility:.2f} W:{wrist.visibility:.2f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                    )
            else:
                cv2.putText(frame, "NO POSE DETECTED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('ROS 2 Arm Publisher', frame)
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
    node = ArmTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()