
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cpp_solver.srv import ExplorationButton

class ExplorationButtonServer(Node):
    def __init__(self):
        super().__init__('exploration_button_server')
        self.srv = self.create_service(ExplorationButton, 'exploration_button', self.handle_exploration_button)
        self.get_logger().info('Exploration button server ready.')

    def handle_exploration_button(self, request, response):
        self.get_logger().info(f'Received exploration button request: {request.button_pressed}')
        response.success = True
        response.message = 'Button press acknowledged'
        return response

def main(args=None):
    rclpy.init(args=args)
    server = ExplorationButtonServer()
    
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
