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
        
        # Default home position
        self.home_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.get_logger().info("Arm Controller Active: Translating Human -> Robot...")

    def angle_callback(self, msg):
        human_angle = msg.data
        
        # Map human elbow to robot elbow (approx 0 to 2.5 rad range)
        robot_angle = ((180.0 - human_angle) / 180.0) * 2.5
        robot_angle = max(0.0, min(robot_angle, 2.5))
        
        traj_msg = JointTrajectory()
        
        # THE FIX: Force timestamp to 0. 
        # This tells Gazebo to execute immediately and ignores the sim_time mismatch!
        traj_msg.header.stamp.sec = 0
        traj_msg.header.stamp.nanosec = 0
        traj_msg.header.frame_id = ''
        
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        positions = list(self.home_positions)
        positions[2] = robot_angle  # elbow_joint is index 2
        
        point.positions = positions
        
        # 0.5 seconds for a smooth, controller-accepted motion
        point.time_from_start = Duration(sec=0, nanosec=500_000_000) 
        
        traj_msg.points = [point]
        self.publisher_.publish(traj_msg)

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
