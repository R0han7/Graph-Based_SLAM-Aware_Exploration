
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import math
from typing import List, Tuple, Dict, Set, Optional

from nav_msgs.msg import OccupancyGrid
from cpp_solver.msg import EdgeDistance
from cpp_solver.srv import RequestGraph

class AStar:
    def __init__(self):
        self.directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        self.direction_costs = [14, 10, 14, 10, 10, 14, 10, 14]
    
    def set_world_data(self, width: int, height: int, data: List[int]):
        self.width = width
        self.height = height
        self.data = data
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path using A* algorithm"""
        if self.is_collision(start) or self.is_collision(goal):
            return []
        
        open_set = []
        closed_set = set()
        g_score = {}
        f_score = {}
        came_from = {}
        
        g_score[start] = 0
        f_score[start] = self.heuristic(start, goal)
        open_set.append((f_score[start], start))
        
        while open_set:
            open_set.sort()
            current_f, current = open_set.pop(0)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            for i, direction in enumerate(self.directions):
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                if self.is_collision(neighbor) or neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self.direction_costs[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        return []
    
    def is_collision(self, pos: Tuple[int, int]) -> bool:
        """Check if position is collision"""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        index = y * self.width + x
        if index >= len(self.data):
            return True
        
        return self.data[index] >= 60 or self.data[index] < 0
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class UpdateDistance(Node):
    def __init__(self):
        super().__init__('update_distance')
        
        # Parameters
        self.declare_parameter('robot_start_position', '0.0 0.0 0.0')
        robot_pos_str = self.get_parameter('robot_start_position').value
        self.robot_init_position = [float(x) for x in robot_pos_str.split()]
        
        # Initialize variables
        self.vertices_position = {}
        self.vertices = []
        self.edge_set = set()
        self.get_prior_map = False
        self.updated_edges = set()
        
        # Directions for checking neighbors
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.pub_distance = self.create_publisher(EdgeDistance, '/edge_distance', qos_profile)
        
        # Subscribers
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.occupancy_grid_callback, qos_profile)
        
        # Service clients
        self.client_prior_graph = self.create_client(RequestGraph, 'prior_graph_service')
        
        # Initialize A* pathfinder
        self.astar_finder = AStar()
        
        # Request prior graph
        self.request_prior_graph()
        
        self.get_logger().info(f"Get robot position: {self.robot_init_position[0]:.2f}, {self.robot_init_position[1]:.2f}")
    
    def request_prior_graph(self):
        """Request prior graph from service"""
        self.get_logger().info("Waiting for prior_graph_service...")
        
        if not self.client_prior_graph.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("prior_graph_service not available.")
            return
        
        request = RequestGraph.Request()
        future = self.client_prior_graph.call_async(request)
        
        def response_callback(future):
            try:
                response = future.result()
                self.vertices = response.vertices
                x_coords = response.x_coords
                y_coords = response.y_coords
                edges_start = response.edges_start
                edges_end = response.edges_end
                
                self.get_logger().info("Receive response of service prior_graph_service.")
                self.get_logger().info(f"Prior map has {len(self.vertices)} vertices, {len(edges_start)} edges")
                
                # Store vertex positions
                for i, vertex_id in enumerate(self.vertices):
                    self.vertices_position[vertex_id] = (x_coords[i], y_coords[i])
                
                # Store edges
                for i in range(len(edges_start)):
                    self.edge_set.add((edges_start[i], edges_end[i]))
                
                self.get_prior_map = True
                
            except Exception as e:
                self.get_logger().error(f"Failed to call service: prior_graph_service - {e}")
        
        future.add_done_callback(response_callback)
    
    def occupancy_grid_callback(self, msg: OccupancyGrid):
        """Process occupancy grid and update edge distances"""
        self.get_logger().info("Received an OccupancyGrid message!")
        self.get_logger().info(f"Width: {msg.info.width}, Height: {msg.info.height}, Size: {len(msg.data)}")
        
        origin_point = msg.info.origin
        self.get_logger().info(f"Map origin x: {origin_point.position.x:.3f}, y: {origin_point.position.y:.3f}")
        
        if not self.get_prior_map:
            self.get_logger().warn("Do not get prior map yet.")
            return
        
        # Setup A* pathfinder
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        
        # Convert data to proper format
        map_data = list(msg.data)
        self.astar_finder.set_world_data(width, height, map_data)
        
        # Find free vertices
        free_vertices = []
        free_vertices_index = []
        
        for vertex_id in self.vertices:
            if vertex_id not in self.vertices_position:
                continue
            
            vertex_pos = self.vertices_position[vertex_id]
            v_map_x = round((vertex_pos[0] - origin_point.position.x) / resolution)
            v_map_y = round((vertex_pos[1] - origin_point.position.y) / resolution)
            
            # Check bounds
            if v_map_x >= width or v_map_x < 0 or v_map_y >= height or v_map_y <= 0:
                continue
            
            # Check if vertex is free
            index = v_map_x + max(0, v_map_y - 1) * width
            if index >= len(map_data) or map_data[index] < 0 or map_data[index] > 60:
                continue
            
            # Check if all neighbors are free
            count_free_neighbor = 0
            for dx, dy in self.directions:
                neighbor_x = v_map_x + dx
                neighbor_y = v_map_y + dy
                
                if neighbor_x >= width or neighbor_x < 0 or neighbor_y >= height or neighbor_y <= 0:
                    continue
                
                neighbor_index = neighbor_x + max(0, neighbor_y - 1) * width
                if neighbor_index < len(map_data) and map_data[neighbor_index] == 0:
                    count_free_neighbor += 1
            
            if count_free_neighbor == 8:
                free_vertices.append(vertex_id)
                free_vertices_index.append((v_map_x, v_map_y))
        
        self.get_logger().info(f"Free vertices size: {len(free_vertices)}")
        
        if len(free_vertices) <= 1:
            return
        
        # Update edge weights
        updated_count = 0
        for i in range(len(free_vertices)):
            for j in range(i):
                v1, v2 = free_vertices[i], free_vertices[j]
                candidate = (v1, v2)
                candidate_reverse = (v2, v1)
                
                # Skip if edge already exists or already updated
                if (candidate in self.edge_set or candidate_reverse in self.edge_set or
                    candidate in self.updated_edges or candidate_reverse in self.updated_edges):
                    continue
                
                # Find A* path
                start_pos = free_vertices_index[i]
                goal_pos = free_vertices_index[j]
                astar_path = self.astar_finder.find_path(start_pos, goal_pos)
                
                if not astar_path:
                    continue
                
                # Calculate accurate distance
                astar_dist = 0.0
                prev_x, prev_y = -1, -1
                
                for point in astar_path:
                    if prev_x >= 0:
                        if abs(point[0] - prev_x) + abs(point[1] - prev_y) > 1:
                            astar_dist += 1.414 * resolution
                        else:
                            astar_dist += resolution
                    prev_x, prev_y = point
                
                # Publish edge distance
                edge_distance = EdgeDistance()
                edge_distance.vertex1 = v1
                edge_distance.vertex2 = v2
                edge_distance.distance = astar_dist
                self.pub_distance.publish(edge_distance)
                
                self.updated_edges.add(candidate)
                updated_count += 1
        
        self.get_logger().info(f"{updated_count} edge distances updated")

def main(args=None):
    rclpy.init(args=args)
    node = UpdateDistance()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
