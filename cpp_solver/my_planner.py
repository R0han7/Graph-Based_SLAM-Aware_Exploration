
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import math
from collections import defaultdict
import time
from typing import List, Tuple, Dict, Set, Optional

from std_msgs.msg import Bool, Int16
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from cpp_solver.srv import TspPathList, RequestGraph, ReliableLoop
from cpp_solver.msg import EdgeDistance

class GridMap:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.data = occupancy_grid.data
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x = occupancy_grid.info.origin.position.x
        self.origin_y = occupancy_grid.info.origin.position.y
        
    def get_size(self) -> int:
        return len(self.data)
    
    def get_width(self) -> int:
        return self.width
    
    def get_height(self) -> int:
        return self.height
    
    def get_resolution(self) -> float:
        return self.resolution
    
    def get_origin_x(self) -> float:
        return self.origin_x
    
    def get_origin_y(self) -> float:
        return self.origin_y
    
    def get_coordinates(self, index: int) -> Tuple[int, int]:
        x = index % self.width
        y = index // self.width
        return x, y
    
    def get_index(self, x: int, y: int) -> int:
        return y * self.width + x
    
    def is_free(self, index: int) -> bool:
        if index < 0 or index >= len(self.data):
            return False
        return self.data[index] == 0
    
    def is_occupied(self, index: int) -> bool:
        if index < 0 or index >= len(self.data):
            return False
        return self.data[index] == 100
    
    def is_unknown(self, index: int) -> bool:
        if index < 0 or index >= len(self.data):
            return False
        return self.data[index] == -1
    
    def is_frontier(self, index: int) -> bool:
        if not self.is_free(index):
            return False
        
        x, y = self.get_coordinates(index)
        
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_index = self.get_index(nx, ny)
                    if self.is_unknown(neighbor_index):
                        return True
        return False
    
    def get_num_free_neighbors(self, index: int) -> int:
        x, y = self.get_coordinates(index)
        count = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_index = self.get_index(nx, ny)
                    if self.is_free(neighbor_index):
                        count += 1
        return count
    
    def get_lethal_cost(self) -> int:
        return 100

class MyPlanner(Node):
    def __init__(self):
        super().__init__('my_planner')
        
        # Parameters
        self.declare_parameter('robot_id', 1)
        self.declare_parameter('min_target_area_size', 10.0)
        self.declare_parameter('visualize_frontiers', False)
        self.declare_parameter('use_local_planner', False)
        self.declare_parameter('save_path_plan_time', ' ')
        
        self.robot_id = self.get_parameter('robot_id').value
        self.min_target_area_size = self.get_parameter('min_target_area_size').value
        self.visualize_frontiers = self.get_parameter('visualize_frontiers').value
        self.use_local_replanning = self.get_parameter('use_local_planner').value
        self.save_plan_time_path = self.get_parameter('save_path_plan_time').value
        
        # Publishers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        if self.visualize_frontiers:
            self.frontier_publisher = self.create_publisher(Marker, 'frontiers', qos_profile)
            self.frontier_to_vertex_publisher = self.create_publisher(Marker, 'frontier_to_vertex', qos_profile)
        
        self.goal_publisher = self.create_publisher(Marker, 'nav_goal', qos_profile)
        self.pub_prior_map = self.create_publisher(MarkerArray, 'prior_map', qos_profile)
        self.pub_tsp_path = self.create_publisher(Marker, 'visual_tsp_path', qos_profile)
        self.pub_tsp_index = self.create_publisher(MarkerArray, 'tsp_index', qos_profile)
        self.debug_path_finder = self.create_publisher(Marker, 'path_finder', qos_profile)
        self.pub_distance_map = self.create_publisher(OccupancyGrid, 'dist_map', qos_profile)
        self.finished_pub = self.create_publisher(Bool, '/finished_topic', qos_profile)
        self.request_replan = self.create_publisher(Int16, 'request_replan', qos_profile)
        
        # Subscribers
        self.sub_stop_exploration = self.create_subscription(
            Bool, 'stop_exploration', self.handle_stop_exploration, qos_profile)
        
        # Service clients
        self.client_prior_graph = self.create_client(RequestGraph, 'prior_graph_service')
        self.client_tsp_plan = self.create_client(TspPathList, 'path_plan_service')
        self.client_reliable_loop = self.create_client(ReliableLoop, 'reliable_loop_service')
        
        # Initialize variables
        self.frontiers = []
        self.frontier_cells = 0
        self.plan = None
        self.offset = [-1, 1, -self.width if hasattr(self, 'width') else -100, 
                     self.width if hasattr(self, 'width') else 100,
                     -self.width-1 if hasattr(self, 'width') else -101,
                     -self.width+1 if hasattr(self, 'width') else -99,
                     self.width-1 if hasattr(self, 'width') else 99,
                     self.width+1 if hasattr(self, 'width') else 101]
        
        self.prior_graph = {}
        self.node_list = []
        self.edge_list = []
        self.center_of_frontier = []
        self.tsp_path = []
        self.is_loop = []
        self.curr_goal_idx = 0
        self.exploration_button = False
        self.loop_closing = False
        self.closing_path = []
        self.curr_closing_idx = 0
        self.reach_goal = False
        self.covered_vertices = set()
        self.plan_time_record = 0.0
        self.unique_markers_id = 0
        
        # Initialize services
        self.initialize_services()
        
        self.get_logger().info("MyPlanner is initialized and running!!")
    
    def initialize_services(self):
        """Initialize service clients and get initial data"""
        # Wait for prior graph service
        self.get_logger().info("Waiting for prior_graph_service...")
        if self.client_prior_graph.wait_for_service(timeout_sec=30.0):
            self.request_prior_graph()
        else:
            self.get_logger().error("prior_graph_service not available.")
        
        # Wait for path plan service
        self.get_logger().info("Waiting for path_plan_service...")
        if self.client_tsp_plan.wait_for_service(timeout_sec=30.0):
            self.request_initial_tsp_plan()
        else:
            self.get_logger().error("path_plan_service not available.")
        
        # Wait for reliable loop service
        self.get_logger().info("Waiting for reliable_loop_service...")
        if self.client_reliable_loop.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("reliable_loop_service is available.")
        else:
            self.get_logger().error("reliable_loop_service not available.")
    
    def request_prior_graph(self):
        """Request prior graph from service"""
        request = RequestGraph.Request()
        future = self.client_prior_graph.call_async(request)
        
        def response_callback(future):
            try:
                response = future.result()
                self.read_prior_map(response.vertices, response.x_coords, response.y_coords,
                                  response.edges_start, response.edges_end)
                self.get_logger().info("Load prior_map.")
                self.draw_prior_graph()
                self.get_logger().info("Draw prior_map in RVIZ.")
            except Exception as e:
                self.get_logger().error(f"Failed to call service: prior_graph_service - {e}")
        
        future.add_done_callback(response_callback)
    
    def request_initial_tsp_plan(self):
        """Request initial TSP plan from service"""
        request = TspPathList.Request()
        request.curr_vertex_idx = 0
        request.covered_vertices = []
        
        future = self.client_tsp_plan.call_async(request)
        
        def response_callback(future):
            try:
                response = future.result()
                self.tsp_path = response.response
                self.is_loop = response.is_loop
                self.get_logger().info(f"Receive tsp path: {self.tsp_path}")
                self.get_logger().info(f"Receive loop index: {self.is_loop}")
                self.publish_tsp_path()
            except Exception as e:
                self.get_logger().error(f"Failed to call service: path_plan_service - {e}")
        
        future.add_done_callback(response_callback)
    
    def handle_stop_exploration(self, msg: Bool):
        """Handle stop exploration button"""
        self.exploration_button = msg.data
    
    def find_exploration_target(self, map_data: GridMap, start: int) -> Tuple[int, int]:
        """Main exploration target finding function"""
        start_time = time.time()
        
        # Remove small unknown cells
        self.remove_small_unknown_cells(map_data)
        
        # Find frontiers
        self.frontiers = []
        self.frontier_cells = 0
        self.find_frontiers(map_data, start)
        
        # Remove small frontiers
        self.remove_small_frontiers()
        
        # Allocate frontiers to vertices
        frontier_to_vertex = [-1] * len(self.frontiers)
        vertex_to_frontiers = defaultdict(list)
        self.allocate_frontier_to_vertex(map_data, frontier_to_vertex, vertex_to_frontiers)
        
        # Publish frontiers
        self.publish_frontier(map_data, frontier_to_vertex)
        
        # Handle exploration button
        if self.exploration_button:
            self.get_logger().warn("Waiting to start...")
            self.plan_time_record = time.time() - start_time
            return start, 1  # EXPL_WAITING
        
        # Handle loop closing
        if self.loop_closing:
            goal = self.perform_reliable_looping(map_data, start)
            if goal is not None:
                self.publish_goal_to_rviz(map_data, goal)
                self.plan_time_record = time.time() - start_time
                return goal, 2  # EXPL_TARGET_SET
        
        # Check if exploration is finished
        if self.curr_goal_idx >= len(self.tsp_path):
            self.publish_finished()
            self.plan_time_record = time.time() - start_time
            return start, 3  # EXPL_FINISHED
        
        # Get current goal
        curr_goal = self.tsp_path[self.curr_goal_idx]
        
        # Check if goal is loop vertex
        if self.is_loop[self.curr_goal_idx]:
            self.get_logger().info(f"Next target is loop vertex {curr_goal}(L)")
            self.set_reliable_loop_path(curr_goal)
            self.loop_closing = True
            self.plan_time_record = time.time() - start_time
            return start, 1  # EXPL_WAITING
        
        # Check if reached goal
        if not self.reach_goal:
            if self.has_reach_vertex_region(map_data, start, curr_goal):
                self.reach_goal = True
                self.get_logger().info(f"Reach vertex {curr_goal}")
            else:
                goal = self.get_waypoint_to_goal(map_data, curr_goal, start)
                self.publish_goal_to_rviz(map_data, goal)
                self.plan_time_record = time.time() - start_time
                return goal, 2  # EXPL_TARGET_SET
        
        # Find best frontier
        if curr_goal in vertex_to_frontiers:
            best_frontier = self.find_distance_to_frontiers(map_data, start, vertex_to_frontiers[curr_goal])
            if best_frontier >= 0:
                center = self.center_of_frontier[best_frontier]
                goal = map_data.get_index(center[0], center[1])
                self.publish_goal_to_rviz(map_data, goal)
                self.plan_time_record = time.time() - start_time
                return goal, 2  # EXPL_TARGET_SET
        
        # Move to next goal
        self.curr_goal_idx += 1
        if self.curr_goal_idx >= len(self.tsp_path):
            self.publish_finished()
            self.plan_time_record = time.time() - start_time
            return start, 3  # EXPL_FINISHED
        
        next_goal = self.tsp_path[self.curr_goal_idx]
        goal = self.get_waypoint_to_goal(map_data, next_goal, start)
        self.reach_goal = False
        self.publish_goal_to_rviz(map_data, goal)
        self.plan_time_record = time.time() - start_time
        return goal, 2  # EXPL_TARGET_SET
    
    def find_frontiers(self, map_data: GridMap, start: int):
        """Find frontiers using wavefront propagation"""
        map_size = map_data.get_size()
        self.plan = [-1.0] * map_size
        
        # Update offsets based on map width
        self.offset = [-1, 1, -map_data.get_width(), map_data.get_width(),
                      -map_data.get_width()-1, -map_data.get_width()+1,
                      map_data.get_width()-1, map_data.get_width()+1]
        
        # Initialize queue with start position
        queue = [(0.0, start)]
        self.plan[start] = 0.0
        
        while queue:
            queue.sort(key=lambda x: x[0])
            distance, index = queue.pop(0)
            
            # Check 4-connected neighbors
            for i in range(4):
                neighbor = index + self.offset[i]
                if (neighbor >= 0 and neighbor < map_size and 
                    self.plan[neighbor] == -1 and map_data.is_free(neighbor)):
                    
                    if map_data.is_frontier(neighbor):
                        self.find_cluster(map_data, neighbor)
                    else:
                        queue.append((distance + map_data.get_resolution(), neighbor))
                    
                    self.plan[neighbor] = distance + map_data.get_resolution()
    
    def find_cluster(self, map_data: GridMap, start_cell: int):
        """Find frontier cluster starting from a frontier cell"""
        frontier = []
        queue = [(0.0, start_cell)]
        
        while queue:
            queue.sort(key=lambda x: x[0])
            distance, index = queue.pop(0)
            
            if not map_data.is_frontier(index):
                continue
            
            frontier.append(index)
            self.frontier_cells += 1
            
            # Add adjacent cells to queue
            for i in range(4):
                neighbor = index + self.offset[i]
                if (map_data.is_free(neighbor) and self.plan[neighbor] == -1):
                    self.plan[neighbor] = distance + map_data.get_resolution()
                    queue.append((distance + map_data.get_resolution(), neighbor))
        
        self.frontiers.append(frontier)
    
    def remove_small_unknown_cells(self, map_data: GridMap):
        """Remove small unknown cells surrounded by free space"""
        remove_unknown = []
        for i in range(map_data.get_size()):
            if map_data.is_unknown(i):
                free_neighbors = map_data.get_num_free_neighbors(i)
                if free_neighbors >= 7:
                    remove_unknown.append(i)
        
        # Set these cells to free
        for i in remove_unknown:
            map_data.data[i] = 0
    
    def remove_small_frontiers(self):
        """Remove small frontiers"""
        if not self.frontiers:
            return
        
        frontier_sizes = [len(f) for f in self.frontiers]
        frontier_sizes.sort()
        mid_size = min(frontier_sizes[len(frontier_sizes)//2], 10)
        
        self.frontiers = [f for f in self.frontiers if len(f) >= mid_size]
        
        if not self.frontiers:
            self.get_logger().warn("No Frontiers in the map!")
    
    def allocate_frontier_to_vertex(self, map_data: GridMap, frontier_to_vertex: List[int], 
                                   vertex_to_frontiers: Dict[int, List[int]]):
        """Allocate each frontier to the nearest vertex"""
        self.center_of_frontier = []
        
        for i, frontier in enumerate(self.frontiers):
            # Find center of frontier
            center_x = sum(map_data.get_coordinates(cell)[0] for cell in frontier) // len(frontier)
            center_y = sum(map_data.get_coordinates(cell)[1] for cell in frontier) // len(frontier)
            
            # Find closest cell to center
            best_cell = None
            best_distance = float('inf')
            
            for cell in frontier:
                if map_data.get_num_free_neighbors(cell) < 4:
                    continue
                x, y = map_data.get_coordinates(cell)
                distance = (x - center_x) ** 2 + (y - center_y) ** 2
                if distance < best_distance:
                    best_distance = distance
                    best_cell = cell
            
            if best_cell is not None:
                x, y = map_data.get_coordinates(best_cell)
                self.center_of_frontier.append((x, y))
                
                # Find nearest vertex
                map_x = map_data.get_origin_x() + x * map_data.get_resolution()
                map_y = map_data.get_origin_y() + y * map_data.get_resolution()
                
                best_vertex = None
                best_dist = float('inf')
                
                for vertex_id in self.node_list:
                    if vertex_id in self.prior_graph:
                        vertex_pos = self.prior_graph[vertex_id]
                        dist = abs(vertex_pos[0] - map_x) + abs(vertex_pos[1] - map_y)
                        if dist < best_dist:
                            best_dist = dist
                            best_vertex = vertex_id
                
                if best_vertex is not None:
                    frontier_to_vertex[i] = best_vertex
                    vertex_to_frontiers[best_vertex].append(i)
    
    def find_distance_to_frontiers(self, map_data: GridMap, start: int, 
                                  curr_frontiers: List[int]) -> int:
        """Find closest reachable frontier using BFS"""
        if not curr_frontiers:
            return -1
        
        # Build frontier center map
        frontier_center_to_index = {}
        for k in curr_frontiers:
            if k < len(self.center_of_frontier):
                center = self.center_of_frontier[k]
                center_index = map_data.get_index(center[0], center[1])
                frontier_center_to_index[center_index] = k
        
        # BFS to find closest frontier
        plan = [-1.0] * map_data.get_size()
        queue = [(0.0, start)]
        plan[start] = 0.0
        
        while queue:
            queue.sort(key=lambda x: x[0])
            distance, index = queue.pop(0)
            
            # Check if reached frontier center
            if index in frontier_center_to_index:
                return frontier_center_to_index[index]
            
            # Expand neighbors
            for i in range(8):
                neighbor = index + self.offset[i]
                if (neighbor >= 0 and neighbor < map_data.get_size() and 
                    map_data.data[neighbor] < map_data.get_lethal_cost() and plan[neighbor] < 0):
                    
                    if i < 4:
                        plan[neighbor] = distance + map_data.get_resolution()
                    else:
                        plan[neighbor] = distance + 1.414 * map_data.get_resolution()
                    
                    queue.append((plan[neighbor], neighbor))
        
        return -1
    
    def has_reach_vertex_region(self, map_data: GridMap, start: int, current_goal: int) -> bool:
        """Check if robot is closest to current goal vertex"""
        if current_goal not in self.prior_graph:
            return False
        
        start_x, start_y = map_data.get_coordinates(start)
        start_world_x = map_data.get_origin_x() + start_x * map_data.get_resolution()
        start_world_y = map_data.get_origin_y() + start_y * map_data.get_resolution()
        
        goal_pos = self.prior_graph[current_goal]
        goal_dist = (goal_pos[0] - start_world_x) ** 2 + (goal_pos[1] - start_world_y) ** 2
        
        for vertex_id in self.node_list:
            if vertex_id == current_goal or vertex_id not in self.prior_graph:
                continue
            vertex_pos = self.prior_graph[vertex_id]
            other_dist = (vertex_pos[0] - start_world_x) ** 2 + (vertex_pos[1] - start_world_y) ** 2
            if other_dist < goal_dist:
                return False
        
        return True
    
    def get_waypoint_to_goal(self, map_data: GridMap, current_goal: int, start: int) -> int:
        """Get waypoint towards goal vertex"""
        if current_goal not in self.prior_graph:
            return start
        
        goal_pos = self.prior_graph[current_goal]
        goal_x = int((goal_pos[0] - map_data.get_origin_x()) / map_data.get_resolution())
        goal_y = int((goal_pos[1] - map_data.get_origin_y()) / map_data.get_resolution())
        
        # Check if goal is in map bounds
        if (0 <= goal_x < map_data.get_width() and 0 <= goal_y < map_data.get_height()):
            goal_index = map_data.get_index(goal_x, goal_y)
            if self.plan[goal_index] >= 0:
                return goal_index
        
        # Find closest frontier as intermediate goal
        if self.center_of_frontier:
            best_frontier = 0
            best_dist = float('inf')
            
            for i, center in enumerate(self.center_of_frontier):
                frontier_x = map_data.get_origin_x() + center[0] * map_data.get_resolution()
                frontier_y = map_data.get_origin_y() + center[1] * map_data.get_resolution()
                dist = (goal_pos[0] - frontier_x) ** 2 + (goal_pos[1] - frontier_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_frontier = i
            
            center = self.center_of_frontier[best_frontier]
            return map_data.get_index(center[0], center[1])
        
        return start
    
    def set_reliable_loop_path(self, curr_goal: int):
        """Set reliable loop closing path"""
        if curr_goal not in self.prior_graph:
            return
        
        goal_pos = self.prior_graph[curr_goal]
        self.closing_path = []
        
        # Simple square pattern around goal
        delta_x = [1.5, 0, -1.5, 0]
        delta_y = [0, 1.5, 0, -1.5]
        
        for i in range(4):
            self.closing_path.append((goal_pos[0] + delta_x[i], goal_pos[1] + delta_y[i]))
        
        self.curr_closing_idx = 0
        self.get_logger().info("Set reliable loop closing path.")
    
    def perform_reliable_looping(self, map_data: GridMap, start: int) -> Optional[int]:
        """Perform reliable loop closing"""
        if self.curr_closing_idx >= len(self.closing_path):
            self.loop_closing = False
            self.curr_goal_idx += 1
            self.reach_goal = False
            return None
        
        # Check if reached current closing point
        start_x, start_y = map_data.get_coordinates(start)
        curr_x = map_data.get_origin_x() + start_x * map_data.get_resolution()
        curr_y = map_data.get_origin_y() + start_y * map_data.get_resolution()
        
        target = self.closing_path[self.curr_closing_idx]
        if (target[0] - curr_x) ** 2 + (target[1] - curr_y) ** 2 < 0.3:
            self.curr_closing_idx += 1
        
        if self.curr_closing_idx < len(self.closing_path):
            target = self.closing_path[self.curr_closing_idx]
            goal_x = int((target[0] - map_data.get_origin_x()) / map_data.get_resolution())
            goal_y = int((target[1] - map_data.get_origin_y()) / map_data.get_resolution())
            return map_data.get_index(goal_x, goal_y)
        
        return None
    
    def read_prior_map(self, vertex_list: List[int], x_coords: List[float], y_coords: List[float],
                      edges_start: List[int], edges_end: List[int]):
        """Read prior map from service response"""
        self.node_list = vertex_list
        self.prior_graph = {}
        
        for i, vertex_id in enumerate(vertex_list):
            self.prior_graph[vertex_id] = (x_coords[i], y_coords[i])
        
        self.edge_list = list(zip(edges_start, edges_end))
    
    def draw_prior_graph(self):
        """Draw prior graph in RVIZ"""
        marker_array = MarkerArray()
        
        # Draw vertices
        points = Marker()
        points.header.frame_id = "map"
        points.header.stamp = self.get_clock().now().to_msg()
        points.type = Marker.POINTS
        points.id = self.unique_markers_id
        self.unique_markers_id += 1
        points.scale.x = 0.8
        points.scale.y = 0.8
        points.color.g = 1.0
        points.color.a = 0.8
        
        for vertex_id in self.node_list:
            if vertex_id in self.prior_graph:
                point = Point()
                pos = self.prior_graph[vertex_id]
                point.x = pos[0]
                point.y = pos[1]
                point.z = 0.0
                points.points.append(point)
        
        marker_array.markers.append(points)
        self.pub_prior_map.publish(marker_array)
    
    def publish_tsp_path(self):
        """Publish TSP path visualization"""
        line_list = Marker()
        line_list.header.frame_id = "map"
        line_list.header.stamp = self.get_clock().now().to_msg()
        line_list.type = Marker.LINE_LIST
        line_list.id = self.unique_markers_id
        self.unique_markers_id += 1
        line_list.scale.x = 0.1
        line_list.color.r = 1.0
        line_list.color.a = 0.1
        
        for i in range(len(self.tsp_path) - 1):
            if (self.tsp_path[i] in self.prior_graph and 
                self.tsp_path[i+1] in self.prior_graph):
                
                p1 = Point()
                pos1 = self.prior_graph[self.tsp_path[i]]
                p1.x = pos1[0]
                p1.y = pos1[1]
                p1.z = 0.0
                line_list.points.append(p1)
                
                p2 = Point()
                pos2 = self.prior_graph[self.tsp_path[i+1]]
                p2.x = pos2[0]
                p2.y = pos2[1]
                p2.z = 0.0
                line_list.points.append(p2)
        
        self.pub_tsp_path.publish(line_list)
    
    def publish_goal_to_rviz(self, map_data: GridMap, goal: int):
        """Publish goal marker to RVIZ"""
        x, y = map_data.get_coordinates(goal)
        
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = map_data.get_origin_x() + x * map_data.get_resolution()
        goal_marker.pose.position.y = map_data.get_origin_y() + y * map_data.get_resolution()
        goal_marker.pose.position.z = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 10 * map_data.get_resolution()
        goal_marker.scale.y = 10 * map_data.get_resolution()
        goal_marker.scale.z = 10 * map_data.get_resolution()
        goal_marker.color.a = 0.5
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        
        self.goal_publisher.publish(goal_marker)
    
    def publish_frontier(self, map_data: GridMap, frontier_to_vertex: List[int]):
        """Publish frontier visualization"""
        if not self.visualize_frontiers:
            return
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.position.x = map_data.get_origin_x()
        marker.pose.position.y = map_data.get_origin_y()
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = map_data.get_resolution()
        marker.scale.y = map_data.get_resolution()
        marker.scale.z = map_data.get_resolution()
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        for i, frontier in enumerate(self.frontiers):
            for cell in frontier:
                x, y = map_data.get_coordinates(cell)
                point = Point()
                point.x = x * map_data.get_resolution()
                point.y = y * map_data.get_resolution()
                point.z = 0.0
                marker.points.append(point)
        
        self.frontier_publisher.publish(marker)
    
    def publish_finished(self):
        """Publish exploration finished message"""
        msg = Bool()
        msg.data = True
        self.finished_pub.publish(msg)
        self.get_logger().info("Exploration finished!")

def main(args=None):
    rclpy.init(args=args)
    node = MyPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
