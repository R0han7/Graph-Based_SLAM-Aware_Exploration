
#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('cpp_solver')
    
    # Declare launch arguments
    strategy_arg = DeclareLaunchArgument(
        'strategy',
        default_value='MyPlanner',
        description='Exploration strategy to use'
    )
    
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value='map3/map3',
        description='Map name to use'
    )
    
    robot_position_arg = DeclareLaunchArgument(
        'robot_position',
        default_value='0 0 0',
        description='Robot starting position'
    )
    
    # Launch the path planner node
    path_planner_node = Node(
        package='cpp_solver',
        executable='path_planner.py',
        name='path_planner',
        output='screen',
        parameters=[{
            'strategy': LaunchConfiguration('strategy'),
            'map_name': LaunchConfiguration('map_name'),
            'robot_start_position': LaunchConfiguration('robot_position')
        }]
    )
    
    # Launch the exploration button server
    button_server_node = Node(
        package='cpp_solver',
        executable='exploration_button_server.py',
        name='exploration_button_server',
        output='screen'
    )
    
    # Launch the updateDistance node
    update_distance_node = Node(
        package='cpp_solver',
        executable='update_distance.py',
        name='updateDistance',
        output='screen',
        parameters=[{
            'robot_start_position': LaunchConfiguration('robot_position')
        }]
    )
    
    return LaunchDescription([
        strategy_arg,
        map_name_arg,
        robot_position_arg,
        path_planner_node,
        button_server_node,
        update_distance_node,
    ])
