#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cpp_solver/msg/edge_distance.hpp>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <set>

#include "cpp_solver/RequestGraph.h"
#include "cpp_solver/EdgeDistance.h"

#include "AStar/AStar2.h"

std::map<int, std::pair<float, float>> verticesPosition;
std::vector<int> vertices;
std::set<std::pair<int, int>> edgeSet;
bool getPriorMap;

std::vector<std::pair<int, int>> dir;
std::set<std::pair<int, int>> updated_edges;

std::vector<double> robotInitPosition;
rclcpp::Publisher<cpp_solver::msg::EdgeDistance>::SharedPtr pubDistance;
std::shared_ptr<rclcpp::Node> node;

void occupancyGridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    RCLCPP_INFO(node->get_logger(), "Received an OccupancyGrid message!");
    RCLCPP_INFO(node->get_logger(), "Width: %d, Height: %d, Size: %d", msg->info.width, msg->info.height, msg->data.size());
    geometry_msgs::msg::Pose originPoint = msg->info.origin;
    RCLCPP_INFO(node->get_logger(), "Map origin x: %.3f, y: %.3f", originPoint.position.x, originPoint.position.y);
    
    if(!getPriorMap){
        RCLCPP_WARN(node->get_logger(), "Do not get prior map yet.");
        return;
    }
    // A* path finder
    float resolution = msg->info.resolution;
    int width = msg->info.width, height = msg->info.height;
    std::unique_ptr<int8_t[]> map_arr(new int8_t[msg->data.size()]);
    for(unsigned int i = 0; i < msg->data.size(); i++){
        // Value: -1 for unknown, 0 for free, 100 for occupied
        map_arr[i] = static_cast<int8_t>(msg->data[i]);
    }

    AStar::PathFinder astar_finder;
    astar_finder.setWorldData(static_cast<unsigned int>(msg->info.width), 
                            static_cast<unsigned int>(msg->info.height), 
                            map_arr.get());
    // Get vertices coordindate index in map, and only remain free vertices
    std::vector<int> free_vertices;
    std::vector<std::pair<int, int>> free_vertices_index;
    for(int i = 0; i < verticesPosition.size(); i++){
        int v = vertices[i];
        int v_map_x = std::round((verticesPosition[v].first - originPoint.position.x) / resolution);
        int v_map_y = std::round((verticesPosition[v].second - originPoint.position.y) / resolution);
        
        if(v_map_x >= width || v_map_x < 0 || v_map_y >= height || v_map_y <= 0)
            continue;
        int index = v_map_x + std::max(0, v_map_y-1) * width;

        if(map_arr[index] < 0 || map_arr[index] > 60)
            continue;
        // Check whether neighbors are free
        int count_free_neighbor = 0;
        for(int j = 0; j < dir.size(); j++){
            int neighbor_x = v_map_x + dir[j].first;
            int neighbor_y = v_map_y + dir[j].second;
            if(neighbor_x >= width || neighbor_x < 0 || neighbor_y >= height || neighbor_y <= 0)
                continue;

            int neighbor_index = neighbor_x + std::max(0, neighbor_y-1) * width;
            if(map_arr[neighbor_index] == 0){
                count_free_neighbor++;
            }
            // if(v == 8)
            //     RCLCPP_WARN("cell value: %d", map_arr[neighbor_index]);
        }
        if(count_free_neighbor == 8){
            free_vertices.push_back(v);
            free_vertices_index.push_back(std::make_pair(v_map_x, v_map_y));
        }
        // if(v == 8)
        //     RCLCPP_WARN("free cell number: %d", count_free_neighbor);
    }
    RCLCPP_INFO(node->get_logger(), "Free vertices size: %d", free_vertices.size());

    if(free_vertices.size() > 1){
        // Update edge weight and publish
        AStar::CoordinateList astar_path;
        for(int k = 0; k < free_vertices.size(); k++){
            for(int p = 0; p < k; p++){
                std::pair<int, int> candidate = std::make_pair(free_vertices[k], free_vertices[p]);
                std::pair<int, int> candidate_reverse = std::make_pair(free_vertices[p], free_vertices[k]);
                if(edgeSet.find(candidate) != edgeSet.end() || updated_edges.find(candidate) != updated_edges.end()
                   || edgeSet.find(candidate_reverse) != edgeSet.end() || updated_edges.find(candidate_reverse) != updated_edges.end()){
                    continue;
                }
                astar_path = astar_finder.findPath({free_vertices_index[k].first, free_vertices_index[k].second}, 
                                                {free_vertices_index[p].first, free_vertices_index[p].second});
                if(astar_path.size() == 0)
                    continue;
                // More accurate distance estimation
                int prev_x = -1, prev_y = -1;
                double astarDist = 0;
                for(auto& point: astar_path){
                    if(prev_x >= 0){
                        if(std::abs(point.x - prev_x) + std::abs(point.y - prev_y) > 1){
                            astarDist += 1.414 * resolution;
                        }else{
                            astarDist += resolution;
                        }
                    }
                    prev_x = point.x;
                    prev_y = point.y;
                }
                // FIXME: astarDist should be less than distance on the prior map
                cpp_solver::msg::EdgeDistance edge_distance;
                edge_distance.vertex1 = free_vertices[k];
                edge_distance.vertex2 = free_vertices[p];
                edge_distance.distance = astarDist;
                pubDistance->publish(edge_distance);
                updated_edges.insert(candidate);           
            }
        }
    }
    RCLCPP_INFO(node->get_logger(), "%d edge distance updated", updated_edges.size());
    return;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("updateDistance");

    dir.push_back(std::make_pair(0, 1));
    dir.push_back(std::make_pair(0, -1));
    dir.push_back(std::make_pair(1, 0));
    dir.push_back(std::make_pair(-1, 0));
    dir.push_back(std::make_pair(1, 1));
    dir.push_back(std::make_pair(1, -1));
    dir.push_back(std::make_pair(-1, 1));
    dir.push_back(std::make_pair(-1, -1));

    getPriorMap = false;
	RCLCPP_INFO(node->get_logger(), "Waiting for prior_graph_service ...");

    //auto clientPriorGraph = node->create_client<cpp_solver::srv::RequestGraph>("prior_graph_service");

	// Waiting for service is not implemented yet
	//bool priorGraphServiceAvailable = rclcpp::service::waitForService("prior_graph_service", rclcpp::Duration(30.0));
    //if (!priorGraphServiceAvailable)
    //{
    //    RCLCPP_ERROR(node->get_logger(), "prior_graph_service not available.");
    //}
	//else{
		//rclcpp::Client<cpp_solver::srv::RequestGraph>::SharedPtr clientPriorGraph = node->create_client<cpp_solver::srv::RequestGraph>("prior_graph_service");
		//auto request = std::make_shared<cpp_solver::srv::RequestGraph::Request>();
		//auto result = clientPriorGraph->async_send_request(request);
		//if (rclcpp::spin_until_complete(node,
        //                      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::seconds(30))) ==
        //rclcpp::FutureReturnCode::TIMEOUT)
        //{
        //    RCLCPP_ERROR(node->get_logger(), "prior_graph_service not available.");
		//}
		//else
		//{	
		//	//cpp_solver::RequestGraph::Response res2 = clientMessage2.response;
		//	vertices = result.get()->response.vertices;
		//	std::vector<float> x_coords = result.get()->response.x_coords;
		//	std::vector<float> y_coords = result.get()->response.y_coords;
		//	std::vector<int32_t> edges_start = result.get()->response.edges_start;
		//	std::vector<int32_t> edges_end = result.get()->response.edges_end;
		//	RCLCPP_INFO(node->get_logger(), "Receive response of service prior_graph_service.");
        //    RCLCPP_INFO(node->get_logger(), "Prior map has %d vertices, %d edges", vertices.size(), edges_start.size());
        //    for(int i = 0; i < vertices.size(); i++){
        //        verticesPosition[vertices[i]] = std::make_pair(x_coords[i], y_coords[i]);
        //    }
        //    for(int i = 0; i < edges_start.size(); i++){
        //        edgeSet.insert(std::make_pair(edges_start[i], edges_end[i]));
        //    }
        //    getPriorMap = true;
		//}
		//else
		//{
		//	RCLCPP_ERROR(node->get_logger(), "Failed to call service: prior_graph_service");
		//}
	//}

    // Read the robot_init_position parameter as a string from the parameter server
    std::string robotInitPositionStr;
    node->declare_parameter("robot_start_position", std::string("0.0 0.0 0.0"));
    if (!node->get_parameter("robot_start_position", robotInitPositionStr)) {
        RCLCPP_ERROR(node->get_logger(), "Failed to read robot_init_position parameter.");
        return 1;
    }

    // Parse the string to get three double values
    std::istringstream iss(robotInitPositionStr);
    double value;
    while (iss >> value) {
        robotInitPosition.push_back(value);
        // Check if there are exactly three values in the string
        if (robotInitPosition.size() == 3) {
            break;
        }
    }

    // Check if there are exactly three values in the string
    if (robotInitPosition.size() != 3) {
        RCLCPP_ERROR(node->get_logger(), "Invalid robot_init_position parameter format. Expected three double values separated by spaces.");
        return 1;
    }
    RCLCPP_INFO(node->get_logger(), "Get robot position: %.2f, %.2f", robotInitPosition[0], robotInitPosition[1]);

    // Create subscriber and publisher
    auto sub = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", rclcpp::QoS(1), occupancyGridCallback);

    pubDistance = node->create_publisher<cpp_solver::msg::EdgeDistance>("/edge_distance", 10);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}