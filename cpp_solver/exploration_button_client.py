
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cpp_solver.srv import ExplorationButton

class ExplorationButtonClient(Node):
    def __init__(self):
        super().__init__('exploration_button_client')
        self.cli = self.create_client(ExplorationButton, 'exploration_button')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def send_request(self, button_pressed):
        request = ExplorationButton.Request()
        request.button_pressed = button_pressed
        future = self.cli.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)
    client = ExplorationButtonClient()
    
    future = client.send_request(True)
    rclpy.spin_until_future_complete(client, future)
    
    if future.result() is not None:
        response = future.result()
        client.get_logger().info(f'Result: {response.success}, {response.message}')
    else:
        client.get_logger().error('Service call failed')
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
