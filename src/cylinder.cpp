#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

/**
 * @brief CylinderDetector class that handles detection of 30cm diameter cylinders from laser scan data
 *        and draws them on a occupancy grid map using OpenCV for image processing. Needs to have a base 
 *        map published to the topic "/map" to draw on and publishes the modified map to the topic "modified_map"
 */
class CylinderDetector : public rclcpp::Node
{
public:
    /**
     * @brief Constructor for CylinderDetector
     */
    CylinderDetector() 
        : Node("cylinder_detector"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
    {
        laser_scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&CylinderDetector::scanCallback, this, std::placeholders::_1));
        
        map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&CylinderDetector::mapCallback, this, std::placeholders::_1));
        
        map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("modified_map", 10);
        
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1), std::bind(&CylinderDetector::getTransform, this));
    }

private:
    // ROS components
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscription_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    geometry_msgs::msg::TransformStamped transform_stamped_;

    nav_msgs::msg::OccupancyGrid modified_map_;

    /**
     * @brief Gets the transform from base_scan to map frame.
     */
    void getTransform()
    {
        try
        {
            transform_stamped_ = tf_buffer_.lookupTransform(
                "map", "base_scan", tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform 'base_scan' to 'map': %s", ex.what());
        }
    }

    /**
     * @brief Callback function for the Occupancy Grid Map.
     * @param msg Occupancy Grid message.
     */
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        // Copy the map to modified_map_
        if (modified_map_.data.size() != msg->data.size())
        {
            modified_map_ = *msg;
        }
        modified_map_.info = msg->info;
        modified_map_.header = msg->header;

        // Update the map data
        for (size_t i = 0; i < msg->data.size(); i++)
        {
            if (modified_map_.data.at(i) < 100)
            {
                modified_map_.data.at(i) = msg->data.at(i);
            }
        }

        // Publish the modified map
        map_publisher_->publish(modified_map_);
        // RCLCPP_INFO(this->get_logger(), "Published modified occupancy map with a circle.");
    }

    /**
     * @brief Draws a circle on the map.
     * @param data Map data array.
     * @param width Map width in cells.
     * @param center_x Center X of the circle.
     * @param center_y Center Y of the circle.
     * @param radius Radius of the circle in cells.
     */
    void drawCircle(std::vector<int8_t> &data, int width, int center_x, int center_y, int radius)
    {
        int x = radius;
        int y = 0;
        int decision = 1 - radius;
        setCirclePoints(data, width, center_x, center_y, x, y);

        while (y < x)
        {
            y++;
            if (decision < 0)
            {
                decision += 2 * y + 1;
            }
            else
            {
                x--;
                decision += 2 * (y - x) + 1;
            }
            setCirclePoints(data, width, center_x, center_y, y, x);
        }
    }

    /**
     * @brief Marks the points of a circle on the map.
     * @param data Map data array.
     * @param width Map width in cells.
     * @param center_x Center X of the circle.
     * @param center_y Center Y of the circle.
     * @param x X offset for the point.
     * @param y Y offset for the point.
     */
    void setCirclePoints(std::vector<int8_t> &data, int width, int center_x, int center_y, int x, int y)
    {
        std::vector<std::pair<int, int>> points = {
            {center_x + x, center_y + y}, {center_x - x, center_y + y}, 
            {center_x + x, center_y - y}, {center_x - x, center_y - y}, 
            {center_x + y, center_y + x}, {center_x - y, center_y + x}, 
            {center_x + y, center_y - x}, {center_x - y, center_y - x}
        };

        for (const auto &[px, py] : points)
        {
            if (px >= 0 && px < width && py >= 0 && py < width)
            {
                data[py * width + px] = 100; // Mark as occupied
            }
        }
    }

    /**
     * @brief Callback function for LaserScan.
     * @param scan_msg LaserScan message.
     */
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        cv::Mat scan_image = laserScanToImage(scan_msg);
        std::vector<cv::Vec3f> circles = detectCirclesInImage(scan_image);

        if (!circles.empty())
        {
            for (size_t i = 0; i < circles.size(); ++i)
            {
                float x = circles[i][0];
                float y = circles[i][1];
                float radius = circles[i][2];

                double real_x, real_y;
                convertImageToCoordinates(x, y, scan_msg->range_max, scan_image.rows, real_x, real_y);

                geometry_msgs::msg::Point pt = transformToGlobalFrame(real_x, real_y, transform_stamped_.transform);

                RCLCPP_INFO(this->get_logger(), "Cylinder detected at: x = %.2f, y = %.2f, radius = %.2f", pt.x, pt.y, radius);

                // Convert circle to grid cells and draw on map
                float real_radius = radius *(0.15/10);
                drawCircleOnMap(pt.x, pt.y, real_radius);
            }
        }

        displayImage(scan_image);
    }

    /**
     * @brief Converts image coordinates to real-world coordinates.
     * @param img_x Image X coordinate.
     * @param img_y Image Y coordinate.
     * @param max_range Maximum range of the laser scan.
     * @param image_size Image size in pixels.
     * @param real_x Real-world X coordinate (output).
     * @param real_y Real-world Y coordinate (output).
     */
    void convertImageToCoordinates(int img_x, int img_y, double max_range, int image_size, double &real_x, double &real_y)
    {
        double scale = (2.0 * max_range) / image_size;
        real_x = (img_x - image_size / 2) * scale;
        real_y = (img_y - image_size / 2) * scale;
    }

    /**
     * @brief Transforms a point from the robot frame to the global frame.
     * @param x_local X coordinate in robot's frame.
     * @param y_local Y coordinate in robot's frame.
     * @param transform Transform from the robot frame to the map frame.
     * @return Transformed point in the global frame.
     */
    geometry_msgs::msg::Point transformToGlobalFrame(double x_local, double y_local, const geometry_msgs::msg::Transform &transform)
    {
        geometry_msgs::msg::Point pt;
        double theta = quaternionToYaw(transform.rotation);
        pt.x = transform.translation.x + (x_local * std::cos(theta) - y_local * std::sin(theta));
        pt.y = transform.translation.y + (x_local * std::sin(theta) + y_local * std::cos(theta));
        return pt;
    }

    /**
     * @brief Converts quaternion to yaw (theta).
     * @param q Quaternion representing orientation.
     * @return Yaw angle (theta) in radians.
     */
    double quaternionToYaw(const geometry_msgs::msg::Quaternion &q)
    {
        double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    /**
     * @brief Converts the laser scan data into a 2D image.
     * @param scan_msg LaserScan message.
     * @return Image created from the laser scan.
     */
    cv::Mat laserScanToImage(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        int image_size = 500;
        double max_range = scan_msg->range_max;

        cv::Mat image(image_size, image_size, CV_8UC1, cv::Scalar(0));
        double angle = scan_msg->angle_min;
        double angle_increment = scan_msg->angle_increment;

        cv::Point prev_point(-1, -1);
        for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
        {
            double range = scan_msg->ranges[i];
            if (range < scan_msg->range_min || range > max_range)
            {
                angle += angle_increment;
                continue;
            }

            double x = range * std::cos(angle);
            double y = range * std::sin(angle);

            int img_x = static_cast<int>((x / max_range) * image_size / 2 + image_size / 2);
            int img_y = static_cast<int>((y / max_range) * image_size / 2 + image_size / 2);

            cv::Point current_point(img_x, img_y);

            if (img_x >= 0 && img_x < image_size && img_y >= 0 && img_y < image_size)
            {
                image.at<uchar>(img_y, img_x) = 255;
            }

            if (prev_point.x != -1 && prev_point.y != -1)
            {
                double distance = std::hypot(x - prev_point.x, y - prev_point.y);
                if (distance < 0.05)
                {
                    cv::line(image, prev_point, current_point, cv::Scalar(255), 1);
                }
            }
            prev_point = current_point;
            angle += angle_increment;
        }
        return image;
    }

    /**
     * @brief Detects circles in the image using Hough Circle Transform.
     * @param image Input image.
     * @return Vector of detected circles.
     */
    std::vector<cv::Vec3f> detectCirclesInImage(const cv::Mat &image)
    {
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            image, circles, cv::HOUGH_GRADIENT, 1, image.rows / 30, 
            50, 9, 7, 13);
        return circles;
    }

    /**
     * @brief Displays the image with detected circles.
     * @param image Image to be displayed.
     */
    void displayImage(const cv::Mat &image)
    {
        cv::imshow("Laser Scan with Detected Cylinders", image);
        cv::waitKey(1);
    }

    /**
     * @brief Draws a circle on the map.
     * @param x_world World X coordinate.
     * @param y_world World Y coordinate.
     * @param radius Radius of the circle.
     */
    void drawCircleOnMap(double x_world, double y_world, double radius)
    {
        
        int x_in_cells = std::round((x_world - modified_map_.info.origin.position.x) / modified_map_.info.resolution);
        int y_in_cells = std::round((y_world - modified_map_.info.origin.position.y) / modified_map_.info.resolution);
        int radius_in_cells = std::round(radius / modified_map_.info.resolution);
        // Log circle and map parameters for debugging
        RCLCPP_INFO(this->get_logger(), "map origin: (%f, %f) meters", modified_map_.info.origin.position.x,modified_map_.info.origin.position.y );
        RCLCPP_INFO(this->get_logger(), "Map Resolution: %f meters/cell", modified_map_.info.resolution);
        RCLCPP_INFO(this->get_logger(), "Map width: %d cells", modified_map_.info.width);
        RCLCPP_INFO(this->get_logger(), "Map height: %d cells", modified_map_.info.height);
        RCLCPP_INFO(this->get_logger(), "Circle center: (%f, %f) meters", x_world, y_world);
        RCLCPP_INFO(this->get_logger(), "Circle center in grid: (%d, %d)", x_in_cells, y_in_cells);
        RCLCPP_INFO(this->get_logger(), "Circle radius in cells: %d", radius_in_cells);
        drawCircle(modified_map_.data, modified_map_.info.width, x_in_cells, y_in_cells, radius_in_cells);

    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CylinderDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}