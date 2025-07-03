
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from abc import ABC, abstractmethod
from typing import Tuple

class ExplorationPlanner(ABC):
    """Abstract base class for exploration planners"""
    
    @abstractmethod
    def find_exploration_target(self, map_data, start: int) -> Tuple[int, int]:
        """
        Find exploration target
        
        Args:
            map_data: Grid map data
            start: Start position index
            
        Returns:
            Tuple of (goal_index, status)
            status: 1=WAITING, 2=TARGET_SET, 3=FINISHED
        """
        pass

class ExplorationPlannerNode(Node):
    """ROS2 node wrapper for exploration planners"""
    
    def __init__(self, planner: ExplorationPlanner):
        super().__init__('exploration_planner')
        self.planner = planner
        
    def find_target(self, map_data, start: int) -> Tuple[int, int]:
        """Delegate to the planner implementation"""
        return self.planner.find_exploration_target(map_data, start)

# Constants for exploration status
EXPL_WAITING = 1
EXPL_TARGET_SET = 2
EXPL_FINISHED = 3
