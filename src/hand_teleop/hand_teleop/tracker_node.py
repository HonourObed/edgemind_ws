import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import cv2
import mediapipe as mp
import math

class PinchDetectorNode(Node):
    def __init__(self):
        # 1. Initialize the Node with the name 'pinch_detector'
        super().__init__('pinch_detector')
        
        # 2. Create the Publisher. 
        # It broadcasts a standard Boolean (True/False) on the topic '/gripper_cmd'
        self.publisher_ = self.create_publisher(Bool, '/gripper_cmd', 10)
        
        # 3. Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.pinch_threshold = 0.05
        
        # 4. Open the laptop webcam
        self.cap = cv2.VideoCapture(0)
        
        # 5. Create the ROS 2 Timer (Replaces the 'while' loop)
        # 0.05 seconds = 20 Hz (20 frames per second)
        timer_period = 0.05  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Pinch Detector Node is broadcasting on /gripper_cmd...')

    def calculate_distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

    def timer_callback(self):
        """This function automatically fires 20 times a second."""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to grab camera frame')
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # Prepare the ROS Message
        msg = Bool()
        msg.data = False  # Default to False (Gripper Open)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                distance = self.calculate_distance(thumb_tip, index_tip)
                
                # If distance is below threshold, msg.data becomes True (Gripper Closed)
                msg.data = bool(distance < self.pinch_threshold)
                
                # Visual Feedback
                state_text = "GRAB (True)" if msg.data else "OPEN (False)"
                color = (0, 255, 0) if msg.data else (0, 0, 255)
                cv2.putText(frame, f"ROS Publish: {state_text}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 6. Publish the data to the ROS 2 Network!
        self.publisher_.publish(msg)

        # Display the video window
        cv2.imshow('ROS 2 Pinch Publisher', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PinchDetectorNode()
    
    try:
        # rclpy.spin() keeps the node alive and listening for timer events
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        # Safe shutdown procedure
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()