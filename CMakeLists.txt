cmake_minimum_required(VERSION 3.8)
project(cpp_solver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV)
if (OpenCV_FOUND)
  add_definitions(-DOPENCV_LIBRARY_FOUND=12)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseGraph.msg"
  "msg/EdgeDistance.msg"
  "srv/ExplorationButton.srv"
  "srv/TspPathList.srv"
  "srv/RequestGraph.srv"
  "srv/ReliableLoop.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

include_directories(
  include/${PROJECT_NAME}/
)

# Create the exploration plugins library
add_library(CppExplorationPlugins SHARED
  src/MyPlanner.cpp
  src/exploration_plugins.cpp
  src/graph/graph.cpp
)

add_library(gridastar STATIC
  src/AStar/AStar2.cpp
  src/AStar/AStar2.h
)

target_compile_options(gridastar PRIVATE -fPIC)

ament_target_dependencies(CppExplorationPlugins
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  pluginlib
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_msgs
  nav2_core
  nav2_util
  nav2_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(CppExplorationPlugins "${cpp_typesupport_target}" gridastar)

# Create executables
add_executable(pub_path src/pubPathGT.cpp)
ament_target_dependencies(pub_path
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
)

add_executable(updateDistance src/updateDistance.cpp)
ament_target_dependencies(updateDistance
  rclcpp
  nav_msgs
  std_msgs
)
target_link_libraries(updateDistance gridastar)
rosidl_get_typesupport_target(cpp_typesupport_target2 ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(updateDistance "${cpp_typesupport_target2}")

# Install
install(TARGETS CppExplorationPlugins
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS pub_path updateDistance
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/tsp.py
  scripts/exploration_button_client.py
  scripts/exploration_button_server.py
  scripts/path_planner.py
  scripts/utils.py
  scripts/offline_tsp_evaluation.py
  scripts/read_drawio_to_nx.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch param world config msg srv
  DESTINATION share/${PROJECT_NAME}/
)

install(FILES
  exploration.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()