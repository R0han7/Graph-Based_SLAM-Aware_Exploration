
#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import math

class Vertex:
    def __init__(self, vertex_id: int, position: Tuple[float, float] = (0.0, 0.0)):
        self.id = vertex_id
        self.position = position
        self.edges = set()
    
    def add_edge(self, edge):
        self.edges.add(edge)
    
    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def get_entering_edges(self):
        """Get edges that enter this vertex"""
        entering = set()
        for edge in self.edges:
            if edge.to_vertex == self or edge.undirected:
                entering.add(edge)
        return entering
    
    def get_exiting_edges(self):
        """Get edges that exit this vertex"""
        exiting = set()
        for edge in self.edges:
            if edge.from_vertex == self or edge.undirected:
                exiting.add(edge)
        return exiting
    
    def get_reachable_vertices(self):
        """Get vertices reachable from this vertex"""
        reachable = set()
        for edge in self.edges:
            if edge.from_vertex == self:
                reachable.add(edge.to_vertex)
            elif edge.undirected:
                other_vertex = edge.to_vertex if edge.from_vertex == self else edge.from_vertex
                reachable.add(other_vertex)
        return reachable

class Edge:
    def __init__(self, from_vertex: Vertex, to_vertex: Vertex, 
                 undirected: bool = False, edge_id: int = -1, 
                 cost: float = -1.0, capacity: int = float('inf')):
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.undirected = undirected
        self.id = edge_id
        self.cost = cost if cost >= 0 else self.calculate_cost()
        self.capacity = capacity
        self.parent_id = edge_id
        
        # Add this edge to vertices
        from_vertex.add_edge(self)
        to_vertex.add_edge(self)
    
    def calculate_cost(self) -> float:
        """Calculate Euclidean distance between vertices"""
        dx = self.to_vertex.position[0] - self.from_vertex.position[0]
        dy = self.to_vertex.position[1] - self.from_vertex.position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_vertices(self):
        """Get vertices connected by this edge"""
        return {self.from_vertex, self.to_vertex}
    
    def __lt__(self, other):
        return self.id < other.id

class Graph:
    def __init__(self):
        self.vertices = {}  # id -> Vertex
        self.edges = {}     # id -> Edge
        self.vertex_id_count = 0
        self.edge_id_count = 0
    
    def add_vertex(self, vertex_id: int, position: Tuple[float, float] = (0.0, 0.0)) -> Optional[Vertex]:
        """Add vertex to graph"""
        if vertex_id in self.vertices:
            return None
        
        vertex = Vertex(vertex_id, position)
        self.vertices[vertex_id] = vertex
        self.vertex_id_count += 1
        return vertex
    
    def add_edge(self, from_id: int, to_id: int, undirected: bool = False, 
                 cost: float = -1.0, capacity: int = float('inf'), 
                 edge_id: int = -1) -> Optional[Edge]:
        """Add edge to graph"""
        if from_id not in self.vertices or to_id not in self.vertices:
            return None
        
        if edge_id == -1:
            edge_id = self.edge_id_count + 1
        
        if edge_id in self.edges:
            return None
        
        from_vertex = self.vertices[from_id]
        to_vertex = self.vertices[to_id]
        
        edge = Edge(from_vertex, to_vertex, undirected, edge_id, cost, capacity)
        self.edges[edge_id] = edge
        self.edge_id_count += 1
        
        return edge
    
    def get_vertex(self, vertex_id: int) -> Optional[Vertex]:
        """Get vertex by id"""
        return self.vertices.get(vertex_id)
    
    def get_edge(self, edge_id: int) -> Optional[Edge]:
        """Get edge by id"""
        return self.edges.get(edge_id)
    
    def remove_vertex(self, vertex_id: int) -> bool:
        """Remove vertex and all connected edges"""
        if vertex_id not in self.vertices:
            return False
        
        vertex = self.vertices[vertex_id]
        
        # Remove all edges connected to this vertex
        edges_to_remove = list(vertex.edges)
        for edge in edges_to_remove:
            self.remove_edge(edge.id)
        
        # Remove vertex
        del self.vertices[vertex_id]
        return True
    
    def remove_edge(self, edge_id: int) -> bool:
        """Remove edge from graph"""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        
        # Remove edge from vertices
        edge.from_vertex.remove_edge(edge)
        edge.to_vertex.remove_edge(edge)
        
        # Remove edge from graph
        del self.edges[edge_id]
        return True
    
    def get_edges_between_vertices(self, v1_id: int, v2_id: int) -> Set[int]:
        """Get edge IDs between two vertices"""
        if v1_id not in self.vertices or v2_id not in self.vertices:
            return set()
        
        v1 = self.vertices[v1_id]
        v2 = self.vertices[v2_id]
        
        edge_ids = set()
        for edge in v1.edges:
            if edge.from_vertex == v2 or edge.to_vertex == v2:
                edge_ids.add(edge.id)
        
        return edge_ids
    
    def clear(self):
        """Clear all vertices and edges"""
        self.vertices.clear()
        self.edges.clear()
        self.vertex_id_count = 0
        self.edge_id_count = 0
    
    def copy(self) -> 'Graph':
        """Create a copy of the graph"""
        new_graph = Graph()
        
        # Copy vertices
        for vertex_id, vertex in self.vertices.items():
            new_graph.add_vertex(vertex_id, vertex.position)
        
        # Copy edges
        for edge_id, edge in self.edges.items():
            new_graph.add_edge(
                edge.from_vertex.id, edge.to_vertex.id,
                edge.undirected, edge.cost, edge.capacity, edge.id
            )
        
        new_graph.vertex_id_count = self.vertex_id_count
        new_graph.edge_id_count = self.edge_id_count
        
        return new_graph
    
    def get_vertex_count(self) -> int:
        """Get number of vertices"""
        return len(self.vertices)
    
    def get_edge_count(self) -> int:
        """Get number of edges"""
        return len(self.edges)
    
    def get_vertex_ids(self) -> List[int]:
        """Get list of vertex IDs"""
        return list(self.vertices.keys())
    
    def get_edge_ids(self) -> List[int]:
        """Get list of edge IDs"""
        return list(self.edges.keys())
