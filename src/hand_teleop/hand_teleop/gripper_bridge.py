import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

class GripperBridgeNode(Node):
    def __init__(self):
        super().__init__('gripper_bridge')
        self.subscription = self.create_subscription(Bool, '/gripper_cmd', self.gripper_callback, 10)
        
        # NOTE: If your vacuum service has a different name in Gazebo, change it here!
        self.service_name = '/vacuum_gripper/switch'
        self.cli = self.create_client(SetBool, self.service_name)
        
        self.current_state = False
        self.get_logger().info("Gripper Bridge Active: Listening for Finger Pinch...")

    def gripper_callback(self, msg):
        target_state = msg.data
        
        # Only send a command if the pinch state actually changed
        if target_state != self.current_state:
            self.current_state = target_state
            
            if target_state:
                self.get_logger().info("PINCH DETECTED: Engaging Suction!")
            else:
                self.get_logger().info("RELEASE DETECTED: Disengaging Suction!")
                
            self.send_request(target_state)

    def send_request(self, state):
        if not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Service {self.service_name} not available. Is Gazebo running?')
            return
            
        req = SetBool.Request()
        req.data = state
        
        # Async call so we don't freeze the node waiting for Gazebo
        self.future = self.cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = GripperBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
