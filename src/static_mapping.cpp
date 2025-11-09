#include "dyna_ins_remover/common.hpp"
#include "dyna_ins_remover/tictoc.hpp"
#include "dyna_ins_remover/ins_occ_check.hpp"
#include "dyna_ins_remover/ins_belief_update.hpp"
#include "dyna_ins_remover/extract_clusters.hpp"

#include "pcl/search/kdtree.h"

#define PATCHWORK

#ifdef PATCHWORK
#include "dyna_ins_remover/patchwork.hpp"
#endif

#ifdef PATCHWORKPP
#include "dyna_ins_remover/patchworkpp.hpp"
#endif

#ifdef TGS
#include "dyna_ins_remover/tgs.hpp"
#endif

typedef pcl::PointXYZI PointType;

int window_size;

float voxel_range, voxel_azimuth, voxel_polar, 
      grid_range, grid_azimuth, grid_height, 
      min_height, max_height, 
      min_range, max_range,
      voxel_leaf, sim_thres;

bool downsample, pub_ground_truth;

ros::Subscriber sub_cloud_info;
ros::Publisher pub_current_cloud, pub_ground_cloud, pub_non_ground_cloud, 
               pub_static_cloud, pub_dynamic_cloud, pub_cluster_cloud, pub_dynamic_map,
               pub_path, pub_debug_cloud, pub_voxel_cloud;

std::shared_ptr<PubGridMap<jsk_recognition_msgs::PolygonArray>> pub_grid_map;

std_msgs::Header header;

nav_msgs::Path path;

pcl::VoxelGrid<PointType> voxel_grid;

std::deque<pcl::PointCloud<PointType>> point_clouds, ground_cache, non_ground_cahce, ground_clouds, static_clouds, dynamic_clouds;
std::deque<Eigen::Matrix4f> poses_cache, poses;

pcl::PointCloud<PointType> static_map, gt_map;


#ifdef PATCHWORK
    boost::shared_ptr<PatchWork<PointType>> patchWork(new PatchWork<PointType>());
#endif

#ifdef PATCHWORKPP
    boost::shared_ptr<PatchWorkpp<PointType>> patchWorkpp(new PatchWorkpp<PointType>());
#endif

#ifdef TGS
    boost::shared_ptr<travel::TravelGroundSeg<PointType>> travel_ground_seg(new travel::TravelGroundSeg<PointType>());
#endif

void cloud_info_handler(const dyna_ins_remover::CloudInfo::ConstPtr& msg)
{
    int seq = msg->seq.data; 
    ROS_INFO("\033[1;33m----> Receive %d th Point Cloud.\033[0m", seq);

    pcl::PointCloud<PointType>::Ptr    point_cloud(new  pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr point_cloud_ds(new  pcl::PointCloud<PointType>());

    pcl::fromROSMsg(msg->point_cloud, *point_cloud);

    header = msg->header;
    geometry_msgs::Pose pose = msg->pose;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = header.stamp;
    pose_stamped.header.frame_id = "map";
    pose_stamped.pose = pose;
    path.poses.push_back(pose_stamped);
    path.header.stamp = header.stamp;
    path.header.frame_id = "map";
    pub_path.publish(path);

    pcl::PointCloud<PointType> current_cloud, dynamic_cloud, static_cloud;

    if (downsample) 
    {
        voxel_grid.setInputCloud(point_cloud);
        voxel_grid.filter(*point_cloud_ds);
        pcl::copyPointCloud(*point_cloud_ds, current_cloud);
    }
    else 
    {
        pcl::copyPointCloud(*point_cloud, current_cloud);
    }
    // printf("current cloud size:%ld\n", current_cloud.size());
    
    double time_taken;
    pcl::PointCloud<PointType> ground, non_ground;

#ifdef PATCHWORK
    patchWork->estimate_ground(current_cloud, ground, non_ground, time_taken);
#endif

#ifdef PATCHWORKPP
    patchWorkpp->estimate_ground(current_cloud, ground, non_ground, time_taken);
#endif

#ifdef TGS
    travel_ground_seg->estimateGround(current_cloud, ground, non_ground, time_taken);
#endif

    Eigen::Quaternionf rot = Eigen::Quaternionf(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Vector3f pos = Eigen::Vector3f(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = rot.toRotationMatrix();
    trans.block<3, 1>(0, 3) = pos;

    static tf::TransformBroadcaster tfMapToLidar;
    tf::Transform mapToLidar = tf::Transform(tf::Quaternion(rot.x(), rot.y(), rot.z(), rot.w()), tf::Vector3(pos.x(), pos.y(), pos.z()));
    tf::StampedTransform stampedMapToLidar = tf::StampedTransform(mapToLidar, header.stamp, "map", "lidar");
    tfMapToLidar.sendTransform(stampedMapToLidar);

    if (poses_cache.size() < window_size) 
    {
        ground_cache.push_back(ground);
        non_ground_cahce.push_back(non_ground);
        poses_cache.push_back(trans);
    }

    // printf("cache size: %ld\n", poses_cache.size());

    if (poses_cache.size() == window_size) 
    {
        TicToc build_local_map;
        Eigen::Matrix4f trans_inv = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f rot_inv = trans.block<3, 3>(0, 0).transpose();
        trans_inv.block<3, 3>(0, 0) =   rot_inv;
        trans_inv.block<3, 1>(0, 3) = - rot_inv * pos;

        pcl::PointCloud<PointXYZII> local_map;

        for (int i = 0; i < window_size; ++i)
        {
            pcl::PointCloud<PointType> local_non_ground = non_ground_cahce[i];
            int cloud_size = local_non_ground.size();

            if (i < window_size - 1) 
            {
                Eigen::Matrix4f local_trans = trans_inv * poses_cache[i];
                for (int j = 0; j < cloud_size; ++j)
                {
                    PointType p = local_non_ground.points[j];
                    float range = std::sqrt(p.x * p.x + p.y * p.y);
                    Eigen::Vector4f pl(p.x, p.y, p.z, 1.0);
                    Eigen::Vector4f pg = local_trans * pl;
                    PointXYZII pt;
                    pt.x = pg.x();
                    pt.y = pg.y();
                    pt.z = pg.z();
                    pt.intensity = p.intensity;
                    pt.index = i;
                    if (min_range < range && range < max_range) 
                    {
                        local_map.points.push_back(pt);
                    }
                }
            }
            else 
            {
                for (int j = 0; j < cloud_size; ++j)
                {
                    PointType p = local_non_ground.points[j];
                    float range = std::sqrt(p.x * p.x + p.y * p.y);
                    PointXYZII pt;
                    pt.x = p.x;
                    pt.y = p.y;
                    pt.z = p.z;
                    pt.intensity = p.intensity;
                    pt.index = i;
                    if (min_range < range && range < max_range) 
                    {
                        local_map.points.push_back(pt);
                    }
                }
            }
        }
        
        printf("local map size: %ld\n", local_map.size());

        ground_cache.pop_front();
        non_ground_cahce.pop_front();
        poses_cache.pop_front();

        CurvedVoxelClusterExtraction<PointXYZII> cvce;
        cvce.setVoxelResolution(voxel_range, voxel_azimuth, voxel_polar);
        cvce.setInputCloud(local_map);
        TicToc isaa;
        std::vector<std::vector<Voxel>> clusters = cvce.getClusters();
        isaa.toc("instance segmentation and association");

        int cluster_num = clusters.size();
        int grid_map_index = 0;

        for (int i = 0; i < cluster_num; ++i) 
        {
            auto cluster = clusters[i];
            pcl::PointCloud<PointType>::Ptr cluster_cloud(new pcl::PointCloud<PointType>());
            std::vector<pcl::PointCloud<PointType>> ins_clouds;
            ins_clouds.resize(window_size);

            // decouple instance
            for (auto& voxel : cluster) 
            {
                std::vector<int> point_indices = voxel.point_indices;
                for (int& point_index : point_indices) 
                {   
                    auto point = local_map.points[point_index];

                    pcl::PointXYZI p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    p.intensity = point.intensity;
                    cluster_cloud->points.push_back(p);

                    int index = point.index;
                    ins_clouds[index].points.push_back(p);
                }
            }

            // printf("****************************************************\n");

            int count = 0;
            for (auto& ins_cloud : ins_clouds) 
            {
                // printf("ins point num: %ld\n", ins_cloud.size());
                if (0 < ins_cloud.size())
                    ++count;
            }

            pcl::PointXYZI max_p, min_p;
            pcl::getMinMax3D(*cluster_cloud, min_p, max_p);
            Eigen::Vector3f box = max_p.getVector3fMap() - min_p.getVector3fMap();
            float diam = std::sqrt(box.x() * box.x() + box.y() * box.y());

            if (1 < count) 
            {
                if (25.0f < diam || 3.5f < box.z() || 3.5f < max_p.z) 
                {
                    static_cloud += *cluster_cloud;
                }
                else 
                {
                    pcl::PointCloud<pcl::PointXYZI> head_cloud;
                    pcl::PointCloud<pcl::PointXYZI> tail_cloud;
                    int head_index = window_size-1, tail_index = 0;

                    while(true) 
                    {
                        if (0 < ins_clouds[head_index].size())
                        {
                            head_cloud = ins_clouds[head_index];
                            break;
                        }
                        --head_index;
                    }

                    while(true) 
                    {
                        if (0 < ins_clouds[tail_index].size())
                        {
                            tail_cloud = ins_clouds[tail_index];
                            break;
                        }
                        ++tail_index;
                    }

                    if (3 < head_cloud.size() || 3 < tail_cloud.size()) 
                    {
                        float sim = 0;
                        InstanceOccupancyCheck<PointType> ioc;
                        ioc.setResolution(grid_range, grid_azimuth);
                        ioc.setInputCloud(tail_cloud, head_cloud);
                        ioc.computeDescriptor();
                        ioc.computeSimilarity(sim);

                        if (sim < sim_thres) 
                        {
                            dynamic_cloud += head_cloud;

                            if (2 < (head_index - tail_index)) 
                            {
                                // printf("**********************************************\n");
                                // 0.45 0.5374
                                // 0.7 0.4
                                InstanceBeliefUpdate<PointType> ibu;
                                ibu.setResolution(grid_range * 2.0, grid_azimuth * 2.0);
                                ibu.setInputCloud(ins_clouds);
                                ibu.setProStatic(0.4);
                                ibu.setProDynamic(0.7);
                                float pro = ibu.getInsBelief();

                                if (pro < 0.5) 
                                {
                                    static_cloud += head_cloud;
                                }
                                else 
                                {
                                    dynamic_cloud += head_cloud;
                                }

                                jsk_recognition_msgs::PolygonArray grid_map = ibu.getGridMap(header);
                                pub_grid_map->publish(grid_map_index, grid_map);
                                // printf("grid map index: %d, pro: %f\n", grid_map_index, pro);
                                ++grid_map_index;
                            }
                            else 
                            {
                                dynamic_cloud += head_cloud;
                            }
                            // 0.45 0.5374
                            // 0.7 0.4
                            // InstanceBeliefUpdate<PointType> ibu;
                            // ibu.setResolution(grid_range, grid_azimuth);
                            // ibu.setInputCloud(ins_clouds);
                            // ibu.setProStatic(0.3);
                            // ibu.setProDynamic(0.7);
                            // float pro = ibu.getInsBelief();

                            // if (pro < 0.5) 
                            // {
                            //     static_cloud += head_cloud;
                            // }
                            // else 
                            // {
                            //     dynamic_cloud += head_cloud;
                            // }

                            // jsk_recognition_msgs::PolygonArray grid_map = ibu.getGridMap(header);
                            // pub_grid_map->publish(grid_map_index, grid_map);
                            // printf("index: %d\n", grid_map_index);
                            // printf("pro: %f\n", pro);
                            // ++grid_map_index;
                        }
                        else 
                        {
                            static_cloud += head_cloud;
                        }
                    }
                    else 
                    {
                        static_cloud += *cluster_cloud;
                    }

                }
            }
            else 
            {
                // 05, 07     3.0 2.5, 2.0
                // 00, 01, 02 3.0 2.0, 1.0
                if (3.0f < diam || 2.5f < box.z() || 2.0f < max_p.z) 
                {
                    static_cloud += *cluster_cloud;
                }
            }
        }

        if (pub_debug_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 debug_cloud_msg;
            pcl::toROSMsg(local_map, debug_cloud_msg);
            debug_cloud_msg.header = header;
            pub_debug_cloud.publish(debug_cloud_msg);
        }

        if (pub_voxel_cloud.getNumSubscribers() != 0) 
        {
            std::unordered_map<int, Voxel> voxels = cvce.getHashVoxels();
            // printf("voxels size: %ld\n", voxels.size());
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            for (auto iter = voxels.begin(); iter != voxels.end(); ++iter) 
            {
                unsigned int r = int(rand() % 255);
                unsigned int g = int(rand() % 255);
                unsigned int b = int(rand() % 255);

                std::vector<int> point_indices = iter->second.point_indices;

                for (int& point_index : point_indices) 
                {
                    auto& point = local_map.points[point_index];
                    pcl::PointXYZRGB color_point;
                    color_point.x = point.x;
                    color_point.y = point.y;
                    color_point.z = point.z;
                    color_point.r = r;
                    color_point.g = g;
                    color_point.b = b;
                    voxel_cloud->points.push_back(color_point);
                }
            }
            sensor_msgs::PointCloud2 voxel_cloud_msg;
            pcl::toROSMsg(*voxel_cloud, voxel_cloud_msg);
            voxel_cloud_msg.header = header;
            pub_voxel_cloud.publish(voxel_cloud_msg);
        }

        if (pub_cluster_cloud.getNumSubscribers() != 0) 
        {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
            for (auto& cluster : clusters)
            {
                unsigned int r = int(rand() % 255);
                unsigned int g = int(rand() % 255);
                unsigned int b = int(rand() % 255);
                for (Voxel& voxel : cluster)
                {
                    std::vector<int> point_indices = voxel.point_indices;
                    for (int& point_index : point_indices) 
                    {
                        auto& point = local_map.points[point_index];
                        pcl::PointXYZRGBA color_point;
                        color_point.x = point.x;
                        color_point.y = point.y;
                        color_point.z = point.z;
                        color_point.r = r;
                        color_point.g = g;
                        color_point.b = b;
                        color_point.a = 255;
                        color_point.a = (int)((point.index + 1)*255/window_size);
                        cluster_cloud->points.push_back(color_point);
                    }
                }
            }
            sensor_msgs::PointCloud2 cluster_cloud_msg;
            pcl::toROSMsg(*cluster_cloud, cluster_cloud_msg);
            cluster_cloud_msg.header = header;
            pub_cluster_cloud.publish(cluster_cloud_msg);
        }
    }

    if (pub_ground_truth == true) 
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_dynamic_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_static_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        for (auto pt : dynamic_cloud.points) 
        {
            uint32_t label = static_cast<uint32_t>(pt.intensity);
            uint32_t sem_label = label & 0xFFFF;
            uint32_t ins_label = label >> 16;
            bool     is_static = true;

            pcl::PointXYZRGB cpt;
            cpt.x = pt.x;
            cpt.y = pt.y;
            cpt.z = pt.z;
            
            for (int class_num : DYNAMIC_CLASSES) 
            {
                if (sem_label == class_num)
                {
                    cpt.r = 0;
                    cpt.g = 255;
                    cpt.b = 0;                        
                    is_static = false;
                    color_dynamic_cloud->points.push_back(cpt);
                    // printf("true positive!\n");
                }
            }

            if (is_static) 
            {
                cpt.r = 159;
                cpt.g = 159;
                cpt.b = 159;
                color_static_cloud->points.push_back(cpt);
            }
        }

        for (auto pt : static_cloud.points) 
        {
            uint32_t label = static_cast<uint32_t>(pt.intensity);
            uint32_t sem_label = label & 0xFFFF;
            uint32_t ins_label = label >> 16;
            bool     is_static = true;

            pcl::PointXYZRGB cpt;
            cpt.x = pt.x;
            cpt.y = pt.y;
            cpt.z = pt.z;
            
            for (int class_num : DYNAMIC_CLASSES) 
            {
                if (sem_label == class_num)
                {
                    cpt.r = 255;
                    cpt.g = 0;
                    cpt.b = 0;                        
                    is_static = false;
                    color_dynamic_cloud->points.push_back(cpt);
                    printf("false positive!\n");
                }
            }

            if (is_static) 
            {
                cpt.r = 159;
                cpt.g = 159;
                cpt.b = 159; 
                color_static_cloud->points.push_back(cpt);
            }
        }

        if (pub_dynamic_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 dynamic_cloud_msg;
            pcl::toROSMsg(*color_dynamic_cloud, dynamic_cloud_msg);
            dynamic_cloud_msg.header = header;
            pub_dynamic_cloud.publish(dynamic_cloud_msg);
        }

        if (pub_static_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 static_cloud_msg;
            pcl::toROSMsg(*color_static_cloud, static_cloud_msg);
            static_cloud_msg.header = header;
            pub_static_cloud.publish(static_cloud_msg);
        }

        // if (pub_dynamic_cloud.getNumSubscribers() != 0)
        // {
        //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_dynamic_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        //     for (auto pt : dynamic_cloud.points) 
        //     {
        //         uint32_t label = static_cast<uint32_t>(pt.intensity);
        //         uint32_t sem_label = label & 0xFFFF;
        //         uint32_t ins_label = label >> 16;
        //         bool     is_static = true;

        //         pcl::PointXYZRGB cpt;
        //         cpt.x = pt.x;
        //         cpt.y = pt.y;
        //         cpt.z = pt.z;
             
        //         for (int class_num : DYNAMIC_CLASSES) 
        //         {
        //             if (sem_label == class_num)
        //             {
        //                 cpt.r = 0;
        //                 cpt.g = 255;
        //                 cpt.b = 0;                        
        //                 is_static = false;
        //                 // printf("true positive!\n");
        //             }
        //         }

        //         if (is_static) 
        //         {
        //             cpt.r = 159;
        //             cpt.g = 159;
        //             cpt.b = 159; 
        //         }

        //         color_dynamic_cloud->points.push_back(cpt);
        //     }

        //     sensor_msgs::PointCloud2 dynamic_cloud_msg;
        //     pcl::toROSMsg(*color_dynamic_cloud, dynamic_cloud_msg);
        //     dynamic_cloud_msg.header = header;
        //     pub_dynamic_cloud.publish(dynamic_cloud_msg);
        // }

        // if (pub_static_cloud.getNumSubscribers() != 0)
        // {
        //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_static_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        //     for (auto pt : static_cloud.points) 
        //     {
        //         uint32_t label = static_cast<uint32_t>(pt.intensity);
        //         uint32_t sem_label = label & 0xFFFF;
        //         uint32_t ins_label = label >> 16;
        //         bool     is_static = true;

        //         pcl::PointXYZRGB cpt;
        //         cpt.x = pt.x;
        //         cpt.y = pt.y;
        //         cpt.z = pt.z;
             
        //         for (int class_num : DYNAMIC_CLASSES) 
        //         {
        //             if (sem_label == class_num)
        //             {
        //                 cpt.r = 255;
        //                 cpt.g = 0;
        //                 cpt.b = 0;                        
        //                 is_static = false;
        //                 printf("false positive!\n");
        //             }
        //         }

        //         if (is_static) 
        //         {
        //             cpt.r = 159;
        //             cpt.g = 159;
        //             cpt.b = 159; 
        //         }

        //         color_static_cloud->points.push_back(cpt);
        //     }

        //     sensor_msgs::PointCloud2 static_cloud_msg;
        //     pcl::toROSMsg(*color_static_cloud, static_cloud_msg);
        //     static_cloud_msg.header = header;
        //     pub_static_cloud.publish(static_cloud_msg);
        // }
    }
    else 
    {
        if (pub_dynamic_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 dynamic_cloud_msg;
            pcl::toROSMsg(dynamic_cloud, dynamic_cloud_msg);
            dynamic_cloud_msg.header = header;
            pub_dynamic_cloud.publish(dynamic_cloud_msg);
        }

        if (pub_static_cloud.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 static_cloud_msg;
            pcl::toROSMsg(static_cloud, static_cloud_msg);
            static_cloud_msg.header = header;
            pub_static_cloud.publish(static_cloud_msg);
        }
    }
    
    if (pub_current_cloud.getNumSubscribers() != 0)
    {
        sensor_msgs::PointCloud2 current_cloud_msg;
        pcl::toROSMsg(current_cloud, current_cloud_msg);
        current_cloud_msg.header = header;
        pub_current_cloud.publish(current_cloud_msg);
    }

    if (pub_ground_cloud.getNumSubscribers() != 0)
    {
        sensor_msgs::PointCloud2 ground_msg;
        pcl::toROSMsg(ground, ground_msg);
        ground_msg.header = header;
        pub_ground_cloud.publish(ground_msg);
    }

    if (pub_non_ground_cloud.getNumSubscribers() != 0)
    {
        sensor_msgs::PointCloud2 non_ground_msg;
        pcl::toROSMsg(non_ground, non_ground_msg);
        non_ground_msg.header = header;
        pub_non_ground_cloud.publish(non_ground_msg);
    }
}

int main(int argc, char**argv) 
{
    ros::init(argc, argv, "dyna_ins_remover");
    
    ROS_INFO("\033[1;33m----> Scan to Scan.\033[0m");

    ros::NodeHandle nh;

    nh.param<int>("window_size", window_size, 3);

    nh.param<float>("voxel_range",   voxel_range,   0.5);
    nh.param<float>("voxel_azimuth", voxel_azimuth, 0.5);
    nh.param<float>("voxel_polar",   voxel_polar,   0.5);

    nh.param<float>("grid_range",   grid_range,   0.5);
    nh.param<float>("grid_azimuth", grid_azimuth, 0.5);
    nh.param<float>("grid_height",  grid_height,  0.5);

    nh.param<float>("min_range", min_range, 2.7);
    nh.param<float>("max_range", max_range, 80.0);

    nh.param<float>("min_height", min_height, -1.0);
    nh.param<float>("max_height", max_height, 3.0);

    nh.param<float>("voxel_leaf", voxel_leaf, 0.1);
    nh.param<float>("sim_thres", sim_thres, 0.6);

    nh.param<bool>("downsample", downsample, true);
    nh.param<bool>("pub_ground_truth", pub_ground_truth, false);

#ifdef TGS
    travel_ground_seg->setParams();
#endif

    sub_cloud_info     = nh.subscribe<dyna_ins_remover::CloudInfo>("/cloud_info", 5, cloud_info_handler);
    pub_current_cloud  = nh.advertise<sensor_msgs::PointCloud2>("/current_cloud", 5);
    pub_ground_cloud     = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 5);
    pub_non_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/non_ground_cloud", 5);
    pub_cluster_cloud    = nh.advertise<sensor_msgs::PointCloud2>("/cluster_cloud", 5);
    pub_debug_cloud      = nh.advertise<sensor_msgs::PointCloud2>("/debug_cloud", 5);
    pub_voxel_cloud      = nh.advertise<sensor_msgs::PointCloud2>("/voxel_cloud", 5);

    pub_static_cloud   = nh.advertise<sensor_msgs::PointCloud2>("/static_cloud", 5);
    pub_dynamic_cloud  = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_cloud", 5);
    pub_dynamic_map    = nh.advertise<sensor_msgs::PointCloud2>("/front_end/dynamic_map", 5);
   
    pub_path           = nh.advertise<nav_msgs::Path>("/path", 5);

    pub_grid_map.reset(new PubGridMap<jsk_recognition_msgs::PolygonArray>(nh, "/grid_map_"));

    voxel_grid.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);

    ros::spin();

	return 0;
}