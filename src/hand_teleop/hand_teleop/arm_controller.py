import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller')
        
        self.subscription = self.create_subscription(
            Float64, '/elbow_angle', self.angle_callback, 10)
            
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        self.home_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.get_logger().info("Arm Controller Active: Translating Human -> Robot...")

    def angle_callback(self, msg):
        human_angle = msg.data
        
        # Map human elbow angle (0-180 deg) to robot elbow joint range (0 to 2.5 rad)
        # When arm is straight (180 deg), robot elbow is near 0.
        # When arm is fully bent (0 deg), robot elbow is near 2.5 rad.
        robot_angle = ((180.0 - human_angle) / 180.0) * 2.5
        robot_angle = max(0.0, min(robot_angle, 2.5))
        
        traj_msg = JointTrajectory()
        
        # FIX 1: Use current ROS time, not zero.
        # A zero stamp causes joint_trajectory_controller to discard the message
        # because it interprets it as a command from the past.
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        
        # FIX 2: Set frame_id (required by some controller versions)
        traj_msg.header.frame_id = ''
        
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        positions = list(self.home_positions)
        positions[2] = robot_angle  # elbow_joint is index 2
        
        point.positions = positions
        
        # FIX 3: 0.1s (100ms) is too fast for the controller and may be rejected.
        # 0.5 seconds is a safe minimum for smooth, accepted trajectories.
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5 seconds
        
        traj_msg.points = [point]
        self.publisher_.publish(traj_msg)
        
        self.get_logger().info(
            f"Human elbow: {human_angle:.1f}° -> Robot elbow: {math.degrees(robot_angle):.1f}°",
            throttle_duration_sec=1.0
        )

def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()