import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller')
        self.subscription = self.create_subscription(Float64MultiArray, '/hand_joystick', self.joy_callback, 10)
        self.publisher_ = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # A proper Crane posture facing forward
        self.home_state = [0.0, -1.0, 1.57, -1.57, 1.57, 0.0]
        
        self.smoothed_deltas = [0.0, 0.0, 0.0]
        self.alpha = 0.35 # Increased smoothing
        self.startup_phase = True
        self.start_time = 0.0

    def joy_callback(self, msg):
        raw_deltas = msg.data
        if len(raw_deltas) < 3: return
        
        if self.startup_phase:
            if self.start_time == 0.0:
                self.start_time = time.time()
                traj_msg = JointTrajectory()
                traj_msg.joint_names = self.joint_names
                point = JointTrajectoryPoint()
                point.positions = self.home_state
                point.time_from_start = Duration(sec=4, nanosec=0)
                traj_msg.points = [point]
                self.publisher_.publish(traj_msg)
            
            if time.time() - self.start_time < 4.5:
                return 
            else:
                self.startup_phase = False

        for i in range(3):
            self.smoothed_deltas[i] = (self.alpha * raw_deltas[i]) + ((1.0 - self.alpha) * self.smoothed_deltas[i])
            
        dx, dy, dz = self.smoothed_deltas
        
        # 1. Base Pan
        target_base = self.home_state[0] - (dx * 2.5) 
        target_base = max(-1.57, min(target_base, 1.57))
        
        # 2. Shoulder Lift (FIXED INVERSION: Subtracting dy moves arm UP when hand goes UP)
        target_lift = self.home_state[1] - (dy * 2.5)
        target_lift = max(-2.5, min(target_lift, 0.0)) 
        
        # 3. Elbow Extension 
        target_elbow = self.home_state[2] - (dz * 8.0)
        target_elbow = max(0.5, min(target_elbow, 2.6))

        traj_msg = JointTrajectory()
        traj_msg.header.stamp.sec = 0
        traj_msg.header.frame_id = ''
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [target_base, target_lift, target_elbow, -1.57, 1.57, self.home_state[5]]
        point.time_from_start = Duration(sec=0, nanosec=70_000_000) 
        
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
