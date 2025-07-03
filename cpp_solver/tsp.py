
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import networkx as nx
import numpy as np
from tsp_solver.greedy import solve_tsp
from std_msgs.msg import Float64MultiArray, Int32MultiArray, Int16

class TSP_PLANNER(Node):
    def __init__(self):
        super().__init__('tsp_planner')
        self.prior_graph = self.build_prior_graph()
        D = self.get_distance_matrix(self.prior_graph)
        self.path = solve_tsp(D, endpoints = (21, None))  # Given start and end points
        for i in range(len(self.path)):
            self.path[i] += 1            # For index consistence with cpp graph object

        self.create_subscription(Float64MultiArray, "slam_pose_graph", self.handle_cov, 10)
        self.create_subscription(Int16, 'request_replan', self.handle_plan_request, 10)
        self.pubTSP = self.create_publisher(Int32MultiArray, 'tsp_path', 10)

        # Pose graph
        self.vertex_list = []
        self.edge_list = []
        self.worst_cov = np.diag([0.001, 0.001, 1e-6])

        # Index of tsp path after which to adjust
        self.adjust_index = -1

    def build_prior_graph(self, draw = True):
        graph = nx.Graph()
        n = 36
        node_attributes = []
        # Add nodes in form of (node, attribute_dict)
        start_x = -30.0
        start_y = 30.0
        k = 1
        for i in range(6):
            for j in range(6):
                x = start_x + 12 * j
                y =  start_y - 12 * i
                attr = {"pos": (x, y)}
                node_attributes.append((k, attr))
                k += 1
        graph.add_nodes_from(node_attributes)
        return graph

    def get_distance_matrix(self, graph):
        # Implementation placeholder
        n = len(graph.nodes())
        return np.random.rand(n, n)

    def handle_cov(self, msg):
        # Implementation placeholder
        pass

    def handle_plan_request(self, msg):
        # Implementation placeholder
        pass

def main(args=None):
    rclpy.init(args=args)
    tsp_planner = TSP_PLANNER()
    
    try:
        rclpy.spin(tsp_planner)
    except KeyboardInterrupt:
        pass
    finally:
        tsp_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
