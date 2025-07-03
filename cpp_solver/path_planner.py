
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import sys
import math
import networkx as nx
import numpy as np
from functools import partial
from tsp_solver.greedy import solve_tsp
from collections import defaultdict
import time

from std_msgs.msg import Float64MultiArray, Int32, Int16
from geometry_msgs.msg import Point

from cpp_solver.srv import TspPathList, RequestGraph, ReliableLoop
from cpp_solver.msg import PoseGraph, EdgeDistance

# Add python path to header file
path = os.path.abspath(".")
sys.path.insert(0, path + "/src/cpp_solver/scripts")

from .utils import write_nx_graph_to_g2o, \
    add_edge_information_matrix, add_graph_weights_as_dopt, save_data, \
    get_path_length
from . import utils

from .offline_tsp_evaluation import connect_tsp_path, offline_evaluate_tsp_path, get_distance_matrix_for_tsp,\
    get_distance_matrix_for_submap_tsp, offline_iterative_evaluate_tsp_path, concorde_tsp_solver, \
    get_distance_matrix_for_new_tsp_solver, find_last_loop_closure

from .read_drawio_to_nx import build_prior_map_from_drawio

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner')
        
        # Declare parameters
        self.declare_parameter('strategy', 'MyPlanner')
        self.declare_parameter('map_name', 'map3/map3')
        self.declare_parameter('robot_start_position', '0.0 0.0 0.0')
        
        # Get parameters
        self.strategy = self.get_parameter('strategy').get_parameter_value().string_value
        self.map_name = self.get_parameter('map_name').get_parameter_value().string_value
        robot_pos_str = self.get_parameter('robot_start_position').get_parameter_value().string_value
        
        # Parse robot position
        robot_pos = [float(x) for x in robot_pos_str.split()]
        self.robot_start_x = robot_pos[0]
        self.robot_start_y = robot_pos[1]
        self.robot_start_theta = robot_pos[2] if len(robot_pos) > 2 else 0.0
        
        # Initialize other parameters
        self.use_drawio = True
        self.need_noise = False
        self.variance = 0.0
        self.map_width = 60.0
        
        # Set paths
        package_share_dir = '/workspace/share/cpp_solver'
        self.xml_path = os.path.join(package_share_dir, 'world', self.map_name + '.xml')
        self.g2o_save_path = '/workspace/data/'
        
        #******************* Prior Map Construction ******************#
        if self.use_drawio:
            self.prior_graph = self.build_prior_graph_with_xml(need_normalize=False, need_noise=self.need_noise, variance=self.variance)
        else:
            self.prior_graph = self.build_prior_graph()
        self.align_prior_map_with_robot()
        self.get_logger().info('Build prior map.')

        # Make prior map a complete graph
        # self.make_priormap_complete()

        self.update_prior_graph_attributes()

        # Change this path to your folder
        save_data(self.prior_graph, "/workspace/data/prior_map.pickle")

        self.solve_tsp_path()
        self.get_logger().info('Find initial tsp path.')

        self.modify_tsp_path()
        self.get_logger().info('Find modified curr_path and loop index')

        # Create services
        self.tsp_service = self.create_service(TspPathList, 'path_plan_service', self.handle_replanning)
        self.graph_service = self.create_service(RequestGraph, 'prior_graph_service', self.handle_prior_graph)
        self.loop_service = self.create_service(ReliableLoop, 'reliable_loop_service', self.handle_reliable_loop)

        # Create publishers and subscribers as needed
        self.pub_pose_idx = self.create_publisher(Float64MultiArray, 'pose_graph_idx', 2)
        self.create_subscription(PoseGraph, "slam_pose_graph", self.handle_pose_graph, 10)
        self.create_subscription(EdgeDistance, "/edge_distance", self.handle_edge_distance, 10)

        self.get_logger().info('g2o_save_path: {}'.format(self.g2o_save_path))
        self.get_logger().info('Path planner node initialized')

    def build_prior_graph_with_xml(self, need_normalize = False, need_noise=False, variance=0) -> nx.graph:
        """ Node attri:
                name: int, start from 1
                position: tuple(x, y)
            Edge attri:
                weight: float
        """
        graph = build_prior_map_from_drawio(self.xml_path, actual_map_width=self.map_width, need_normalize=need_normalize, need_noise=need_noise, variance=variance)
        self.get_logger().info(f"Build prior map with {len(graph.nodes())} vertices.")
        return graph

    def build_prior_graph(self):
        # Implementation placeholder
        return nx.Graph()

    def align_prior_map_with_robot(self):
        # Implementation placeholder
        pass

    def update_prior_graph_attributes(self):
        # Implementation placeholder
        pass

    def solve_tsp_path(self):
        # Implementation placeholder
        pass

    def modify_tsp_path(self):
        # Implementation placeholder
        pass

    def handle_replanning(self, request, response):
        # Implementation placeholder
        return response

    def handle_prior_graph(self, request, response):
        # Implementation placeholder
        return response

    def handle_reliable_loop(self, request, response):
        # Implementation placeholder
        return response

    def handle_pose_graph(self, msg):
        # Implementation placeholder
        pass

    def handle_edge_distance(self, msg):
        # Implementation placeholder
        pass

def main(args=None):
    rclpy.init(args=args)
    path_planner = PathPlannerNode()
    
    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
