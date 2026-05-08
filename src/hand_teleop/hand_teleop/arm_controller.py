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
        
        robot_angle = ((180.0 - human_angle) / 180.0) * math.pi
        robot_angle = max(0.0, min(robot_angle, 2.5))
        
        traj_msg = JointTrajectory()
        
        # FIXED: Set time to exactly 0 to force immediate execution
        traj_msg.header.stamp.sec = 0
        traj_msg.header.stamp.nanosec = 0
        
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        positions = list(self.home_positions)
        positions[2] = robot_angle  
        
        point.positions = positions
        # Tell the robot to reach the target extremely fast (0.1 seconds)
        point.time_from_start = Duration(sec=0, nanosec=100000000) 
        
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