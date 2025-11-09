#include "dyna_ins_remover/common.hpp"

int main(int argc, char**argv) 
{
    ros::init(argc, argv, "dyna_ins_remover");

    ros::NodeHandle nh;
    ros::Publisher pub_point_cloud      = nh.advertise<sensor_msgs::PointCloud2>("/point_cloud", 5);
    ros::Publisher pub_gt_static_cloud  = nh.advertise<sensor_msgs::PointCloud2>("/gt_static_cloud", 5);
    ros::Publisher pub_gt_dynamic_cloud = nh.advertise<sensor_msgs::PointCloud2>("/gt_dynamic_cloud", 5);
    ros::Publisher pub_cloud_info       = nh.advertise<dyna_ins_remover::CloudInfo>("/cloud_info", 5);

    std_msgs::Header header;
    header.frame_id = "lidar";

    int start_frame, end_frame, bin_num, label_num;
    double pub_frequency, frequency_time, cost_time;
    bool stop_each_frame, pub_ground_truth;
    string data_path, data_type;

    nh.param<string>("data_type", data_type, "");

    nh.param<int>("start_frame", start_frame, 0);
    nh.param<int>("end_frame", end_frame, 100);
    nh.param<double>("pub_frequency", pub_frequency, 10.0);
    nh.param<bool>("stop_each_frame", stop_each_frame, false);
    nh.param<bool>("pub_ground_truth", pub_ground_truth, false);
    nh.param<string>("data_path", data_path, "");

    string velodyne_path = data_path + "/velodyne";
    string labels_path = data_path + "/labels";
    string poses_path = data_path + "/poses.txt";
    string calib_path = data_path + "/calib.txt";

    vector<string> cloud_files, label_files;
    bin_num = getDirList(velodyne_path, cloud_files);
    label_num = getDirList(labels_path, label_files);

    std::vector<Eigen::Matrix4f> poses;
    getPoses(poses_path, calib_path, poses);

    ROS_INFO("\033[1;32m----> Point Cloud Path: %s\033[0m", velodyne_path.c_str());
    ROS_INFO("\033[1;32m----> Labels Path: %s\033[0m", labels_path.c_str());
    ROS_INFO("\033[1;32m----> Poses Path: %s\033[0m", poses_path.c_str());
    ROS_INFO("\033[1;32m----> Calib Path: %s\033[0m", calib_path.c_str());
    ROS_INFO("\033[1;32m----> Scan Num: %d, Label Num: %d\033[0m", bin_num, label_num);

    signal(SIGINT, signalHandler);

    ROS_INFO("\033[1;32m----> Publish Point Cloud Info.\033[0m");

    frequency_time = 1.0 / pub_frequency;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    ros::Duration(1.0).sleep();

    for (int i = max(0, start_frame); i <= min(bin_num -1, end_frame); ++i)
    {           
        start = std::chrono::system_clock::now();
        
        string cloud_path = velodyne_path + "/" + cloud_files[i];
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZI>());

        if (data_type == "bin") 
        {
            if (pub_ground_truth == true) 
            {
                string label_path = labels_path + "/" + label_files[i];
                getPointCloud(cloud_path, label_path, *point_cloud);
            }
            else
                getPointCloud(cloud_path, *point_cloud);
        }

        else 
        {
            pcl::io::loadPCDFile<pcl::PointXYZI> (cloud_path, *point_cloud);
        }
        
        Eigen::Matrix4f T = poses[i];
        Eigen::Matrix3f R = T.block<3,3>(0,0);
        Eigen::Vector3f t = T.block<3,1>(0,3);
        Eigen::Quaternionf Q(R);

        header.stamp = ros::Time::now();

        ROS_INFO("\033[1;32m----> Publish %d th Point Cloud.\033[0m", i);

        static tf::TransformBroadcaster tfMapToLidar;
        tf::Transform mapToLidar = tf::Transform(tf::Quaternion(Q.x(), Q.y(), Q.z(), Q.w()), tf::Vector3(t.x(), t.y(), t.z()));
        tf::StampedTransform stampedMapToLidar = tf::StampedTransform(mapToLidar, header.stamp, "map", "lidar");
        tfMapToLidar.sendTransform(stampedMapToLidar);

        if (pub_cloud_info.getNumSubscribers() != 0) 
        {
            dyna_ins_remover::CloudInfo cloud_info;
            
            cloud_info.seq.data = i;
            cloud_info.header = header;

            sensor_msgs::PointCloud2 point_cloud_msg;
            pcl::toROSMsg(*point_cloud, point_cloud_msg);
            point_cloud_msg.header = header;
            cloud_info.point_cloud = point_cloud_msg;

            geometry_msgs::Pose pose;
            pose.position.x = t.x();
            pose.position.y = t.y();
            pose.position.z = t.z();
            pose.orientation.x = Q.x();
            pose.orientation.y = Q.y();
            pose.orientation.z = Q.z();
            pose.orientation.w = Q.w();
            cloud_info.pose = pose;

            pub_cloud_info.publish(cloud_info);
        }

        if (pub_point_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 current_cloud_msg;
            pcl::toROSMsg(*point_cloud, current_cloud_msg);
            current_cloud_msg.header = header;
            pub_point_cloud.publish(current_cloud_msg);
        }

        if (pub_ground_truth == true) 
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr gt_static_cloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr gt_dynamic_cloud(new pcl::PointCloud<pcl::PointXYZI>());

            for (auto point : point_cloud->points) 
            {
                uint32_t label = static_cast<uint32_t>(point.intensity);
                uint32_t sem_label = label & 0xFFFF;
                uint32_t ins_label = label >> 16;
                bool     is_static = true;

                for (int class_num : DYNAMIC_CLASSES) {
                    if (sem_label == class_num) 
                    {
                        gt_dynamic_cloud->points.push_back(point);
                        is_static = false;
                    }
                }

                if (is_static) 
                {
                    gt_static_cloud->points.push_back(point);
                }
            }

            sensor_msgs::PointCloud2 gt_static_cloud_msg;
            pcl::toROSMsg(*gt_static_cloud, gt_static_cloud_msg);
            gt_static_cloud_msg.header = header;
            pub_gt_static_cloud.publish(gt_static_cloud_msg);

            sensor_msgs::PointCloud2 gt_dynamic_cloud_msg;
            pcl::toROSMsg(*gt_dynamic_cloud, gt_dynamic_cloud_msg);
            gt_dynamic_cloud_msg.header = header;
            pub_gt_dynamic_cloud.publish(gt_dynamic_cloud_msg);
        }

        if (stop_each_frame)
            cin.ignore();
        
        else 
        {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> delta_seconds = end - start;
            cost_time = delta_seconds.count();
            ros::Duration(frequency_time - cost_time).sleep();
        }
    }

	return 0;
}