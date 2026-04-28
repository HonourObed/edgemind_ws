import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

class GripperBridgeNode(Node):
    def __init__(self):
        super().__init__('gripper_bridge')
        
        # 1. State Tracker (This prevents Gazebo spam!)
        self.current_state = False
        
        # 2. Subscribe to your webcam node
        self.subscription = self.create_subscription(
            Bool,
            '/gripper_cmd',
            self.listener_callback,
            10)
            
        # 3. Create a Client to talk to the Gazebo Vacuum Plugin
        self.cli = self.create_client(SetBool, '/ur5e/vacuum_gripper/switch')
        
        self.get_logger().info('Bridge Node Active: Routing Camera to Gazebo...')

    def listener_callback(self, msg):
        # The core logic: Only trigger if the hand state ACTUALLY changed
        if msg.data != self.current_state:
            
            # Check if Gazebo is actually running before sending the command
            if not self.cli.wait_for_service(timeout_sec=0.1):
                self.get_logger().warning('Gazebo not detected. Holding command.')
                return
                
            self.current_state = msg.data
            
            # Formulate the service request
            req = SetBool.Request()
            req.data = msg.data
            
            if msg.data:
                self.get_logger().info('STATE CHANGE: Hand Pinched -> Commanding Gazebo to GRAB')
            else:
                self.get_logger().info('STATE CHANGE: Hand Opened -> Commanding Gazebo to DROP')
                
            # Send the request to Gazebo asynchronously (so it doesn't freeze our node)
            future = self.cli.call_async(req)
            future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        """Checks if Gazebo successfully executed our command."""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warning(f'Gazebo Error: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = GripperBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()  