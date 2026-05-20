import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller')
        self.subscription = self.create_subscription(Float64MultiArray, '/arm_angles', self.angle_callback, 10)
        self.publisher_ = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # THE PERFECT T-POSE (Robot Home State)
        # Pan: 90 deg right. Lift: Horizontal. Elbow: Straight. Wrist 1 & 2: Aligned. Wrist 3: Locked.
        self.home_state = [1.57, 0.0, 0.0, 0.0, 1.57, 0.0]
        
        self.smoothed_deltas = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.alpha = 0.6 # High responsiveness
        
        self.get_logger().info("3D Controller Online: Awaiting Delta Telemetry...")

    def angle_callback(self, msg):
        raw_deltas = msg.data
        
        # Apply smoothing to the Deltas
        for i in range(5):
            self.smoothed_deltas[i] = (self.alpha * raw_deltas[i]) + ((1.0 - self.alpha) * self.smoothed_deltas[i])
            
        d_sh_yaw, d_sh_pitch, d_el_pitch, d_wr_pitch, d_wr_yaw = self.smoothed_deltas
        
        # TARGET = HOME_STATE + (HUMAN_DELTA * Scaling_Factor)
        # Note: We invert certain deltas depending on the camera mirror direction
        
        # 1. Base Pan (Shoulder Yaw)
        target_base = self.home_state[0] - d_sh_yaw 
        target_base = max(0.0, min(target_base, 3.14)) # Constrain to the front 180 degrees
        
        # 2. Shoulder Lift (Shoulder Pitch)
        target_lift = self.home_state[1] + d_sh_pitch
        target_lift = max(-1.57, min(target_lift, 1.57)) # Prevent hitting the floor
        
        # 3. Elbow Pitch
        target_elbow = self.home_state[2] + d_el_pitch
        target_elbow = max(0.0, min(target_elbow, 2.5)) # Standard UR elbow limit
        
        # 4. Wrist 1 (Wrist Pitch)
        target_w1 = self.home_state[3] + d_wr_pitch
        target_w1 = max(-1.57, min(target_w1, 1.57))
        
        # 5. Wrist 2 (Wrist Yaw)
        target_w2 = self.home_state[4] + d_wr_yaw
        target_w2 = max(0.0, min(target_w2, 3.14))

        # Build execution command
        traj_msg = JointTrajectory()
        traj_msg.header.stamp.sec = 0
        traj_msg.header.frame_id = ''
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [target_base, target_lift, target_elbow, target_w1, target_w2, self.home_state[5]]
        point.time_from_start = Duration(sec=0, nanosec=150_000_000)
        
        traj_msg.points = [point]
        self.publisher_.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
