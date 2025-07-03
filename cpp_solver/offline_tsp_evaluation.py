
#!/usr/bin/env python3

import networkx as nx
import numpy as np
import math
from tsp_solver.greedy import solve_tsp
from scipy.spatial import KDTree

def connect_tsp_path():
    """Connect TSP path"""
    pass

def offline_evaluate_tsp_path():
    """Offline evaluate TSP path"""
    pass

def get_distance_matrix_for_tsp():
    """Get distance matrix for TSP"""
    pass

def get_distance_matrix_for_submap_tsp():
    """Get distance matrix for submap TSP"""
    pass

def offline_iterative_evaluate_tsp_path():
    """Offline iterative evaluate TSP path"""
    pass

def concorde_tsp_solver():
    """Concorde TSP solver"""
    pass

def get_distance_matrix_for_new_tsp_solver():
    """Get distance matrix for new TSP solver"""
    pass

def find_last_loop_closure():
    """Find last loop closure"""
    pass

def add_weight_attr_to_graph(graph, add_random=False):
    for edge in graph.edges():
        node1, node2 = edge
        pose1, pose2 = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
        if add_random:
            graph.edges()[edge]["weight"] = 0.6 + random.randint(0, 6) * 0.1
        else:
            dist = math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)
            graph.edges()[edge]["weight"] = dist
