
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cpp_solver.srv import ExplorationButton
from std_msgs.msg import Bool

class ExploreButtonServer(Node):
    def __init__(self):
        super().__init__('exploration_button_server')
        self.pubStop = self.create_publisher(Bool, 'stop_exploration', 10)
        self.srv = self.create_service(ExplorationButton, 'explore_button', self.handle_exploration_button)
        self.get_logger().info('Start exploration_button_server.')

    def handle_exploration_button(self, request, response):
        msg = Bool()
        if request.value:
            msg.data = True
        else:
            msg.data = False
        self.pubStop.publish(msg)
        self.get_logger().info(f'Publish exploration button: {msg.data}')
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    server_node = ExploreButtonServer()
    rclpy.spin(server_node)
    server_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
