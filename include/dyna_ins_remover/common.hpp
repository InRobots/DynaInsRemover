#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <deque>
#include <thread>
#include <mutex>
#include <signal.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/LinearMath/Quaternion.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/esf.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>



#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <dyna_ins_remover/CloudInfo.h>

using namespace std;

struct PosePoint
{
    double x;
    double y;
    double z;
    double qw;
    double qx;
    double qy;
    double qz;
    int index;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PosePoint,
    (double, x, x) (double, y, y) (double, z, z)
    (double, qw, qw) (double, qx, qx) (double, qy, qy) (double, qz, qz)
    (int, index, index) (double, time, time)
)

struct PointXYZII
{
	PCL_ADD_POINT4D;                    
	float intensity;                 
	std::uint16_t index;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZII,
	(float, x, x) (float, y, y) (float, z, z)
	(float, intensity, intensity)
	(std::uint16_t, index, index)
)

std::vector<int> DYNAMIC_CLASSES = {251, 252, 253, 254, 255, 256, 257, 258, 259};

void signalHandler (int signum) 
{
    ROS_INFO("\033[1;32m----> Exit CVSM\033[0m");
    exit(signum);
}

int getDirList(string dir, vector<string> &files)
{

    files.clear();
    vector<string> tmp_files;
    struct dirent **namelist;
    int n;
    n = scandir(dir.c_str(),&namelist, 0 , alphasort);
    if (n < 0)
    {
        string errmsg{(string{"No directory ("} + dir + string{")"})};
        const char * ptr_errmsg = errmsg.c_str();
        perror(ptr_errmsg);
        // perror("No directory");
    }
    else
    {
        while (n--)
        {
            if(string(namelist[n]->d_name) != "." && string(namelist[n]->d_name) != "..")
            {
            tmp_files.push_back(string(namelist[n]->d_name));
            }
            free(namelist[n]);
        }
        free(namelist);
    }

    for(auto iter = tmp_files.rbegin() ; iter!= tmp_files.rend() ; iter++)
    {
        files.push_back(*iter);
    }
    int num = (int)files.size();
    return num;
}

inline std::string trim(const std::string& str, const std::string& whitespaces =" \0\t\n\r\x0B")
{
	int32_t beg = 0;
	int32_t end = 0;

	/** find the beginning **/
	for (beg = 0; beg < (int32_t) str.size(); ++beg)
	{
		bool found = false;
		for (uint32_t i = 0; i < whitespaces.size(); ++i)
		{
			if (str[beg] == whitespaces[i])
			{
				found = true;
				break;
			}
		}
		if (!found) 
			break;
	}

	/** find the end **/
	for (end = int32_t(str.size()) - 1; end > beg; --end)
	{
		bool found = false;
		for (uint32_t i = 0; i < whitespaces.size(); ++i)
		{
			if (str[end] == whitespaces[i])
			{
				found = true;
				break;
			}
		}
		if (!found)
			break;
	}

	return str.substr(beg, end - beg + 1);
}

inline std::vector<std::string> split(const std::string& line, const std::string& delim = " ", bool skipEmpty = false)
{
	std::vector<std::string> tokens;

	boost::char_separator<char> sep(delim.c_str(), "", (skipEmpty ? boost::drop_empty_tokens : boost::keep_empty_tokens));
	boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);

	for (auto it = tokenizer.begin(); it != tokenizer.end(); ++it)
		tokens.push_back(*it);

	return tokens;
}

inline void getPoses(const std::string& poses_filename, std::vector<Eigen::Matrix4f>& poses) {
    std::ifstream fp(poses_filename.c_str());
    std::string line;

    if (!fp.is_open()) throw std::runtime_error("Unable to open pose file :" + poses_filename);

    fp.peek();

    while (fp.good()) 
    {
		Eigen::Matrix4f P = Eigen::Matrix4f::Identity();

		std::getline(fp, line);
		std::vector<std::string> entries = split(line, " ");

		if (entries.size() < 12) {
			fp.peek();
			continue;
		}

      	for (uint32_t i = 0; i < 12; ++i) {
    		P(i / 4, i - int(i / 4) * 4) = boost::lexical_cast<float>(trim(entries[i]));
      	}

      	poses.push_back(P);

      	fp.peek();
    }

    fp.close();
}

inline void getPoses(const std::string& poses_filename, const std::string& calib_filename, std::vector<Eigen::Matrix4f>& poses) 
{

    std::ifstream fp(poses_filename.c_str());
    std::string line;

    if (!fp.is_open()) throw std::runtime_error("Unable to open pose file :" + poses_filename);

    fp.peek();

    while (fp.good()) 
    {
		Eigen::Matrix4f P = Eigen::Matrix4f::Identity();

		std::getline(fp, line);
		std::vector<std::string> entries = split(line, " ");

		if (entries.size() < 12) {
			fp.peek();
			continue;
		}

      	for (uint32_t i = 0; i < 12; ++i) {
    		P(i / 4, i - int(i / 4) * 4) = boost::lexical_cast<float>(trim(entries[i]));
      	}

      	poses.push_back(P);

      	fp.peek();
    }

    fp.close();

    std::map<std::string, Eigen::Matrix4f> calib;
    std::ifstream fc(calib_filename.c_str());
    if (!fc.is_open())
    {
      	throw std::runtime_error(std::string("Unable to open calibration file: ") + calib_filename);
    }

    fc.peek();
    
	while (!fc.eof())
	{
		std::getline(fc, line);
		std::vector<std::string> tokens = split(line, ":");

      	if (tokens.size() == 2) 
		{
      		std::string name = trim(tokens[0]);
      		std::vector<std::string> entries = split(trim(tokens[1]), " ");
			if (entries.size() == 12) 
			{
				Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
				for (uint32_t i = 0; i < 12; ++i) 
				{
					m(i / 4, i - int(i / 4) * 4) = boost::lexical_cast<float>(trim(entries[i]));
				}

				calib[name] = m;
			}
      	}
      	fc.peek();
    }
    fc.close();

    // convert from camera to velodyne coordinate system.
    Eigen::Matrix4f Tr = calib["Tr"];
    Eigen::Matrix4f Tr_inv = Tr.inverse();
    for (int i = 0; i < poses.size(); ++i) 
    {
      poses[i] = Tr_inv * poses[i] * Tr;
    }
}

inline void getPointCloud(const std::string& bin_file, pcl::PointCloud<pcl::PointXYZI> &cloud) 
{	
	FILE *file = fopen(bin_file.c_str(), "rb");
	if (!file)
	{
		throw invalid_argument("Could not open the .bin file!");
	}
	else 
	{
		std::vector<float> buffer(1000000);
		size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
		fclose(file);
		cloud.points.resize(num_points);

		for (int i = 0; i < num_points; i++) 
		{
			auto &pt = cloud.at(i);
			pt.x = buffer[i * 4];
			pt.y = buffer[i * 4 + 1];
			pt.z = buffer[i * 4 + 2];
			pt.intensity = buffer[i * 4 + 3];
		}
	}
}

inline void getLabel(const std::string& label_file, pcl::PointCloud<pcl::PointXYZI> &cloud) 
{
	std::ifstream label_input(label_file, std::ios::binary);
	if (!label_input.is_open())
	{
		throw invalid_argument("Could not open the .label file!");
	}
}

// inline void getPointCloud(const std::string& bin_file, const std::string& label_file, pcl::PointCloud<pcl::PointXYZI> &cloud) 
// {	
// 	FILE *file = fopen(bin_file.c_str(), "rb");
// 	std::ifstream label_input(label_file, std::ios::binary);
// 	if (!file || !label_input.is_open())
// 	{
// 		throw invalid_argument("Could not open the .bin or .label file!");
// 	}
// 	else 
// 	{
// 		std::vector<float> buffer(1000000);
// 		size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
// 		fclose(file);
// 		cloud.points.resize(num_points);
// 		label_input.seekg(0, std::ios::beg);
// 		std::vector<uint32_t> labels(num_points);
// 		label_input.read((char*)&labels[0], num_points * sizeof(uint32_t));

// 		for (int i = 0; i < num_points; i++) 
// 		{
// 			auto &pt = cloud.at(i);
// 			pt.x = buffer[i * 4];
// 			pt.y = buffer[i * 4 + 1];
// 			pt.z = buffer[i * 4 + 2];
// 			std::uint16_t label = labels[i] & 0xFFFF;
// 			if (250 < label) 
// 			{
// 				pt.intensity = buffer[i * 4 + 3] + 250.0;
// 			}
// 			else
// 			{
// 				pt.intensity = buffer[i * 4 + 3];
// 			}
// 		}
// 	}
// }

inline void getPointCloud(const std::string& bin_file, const std::string& label_file, pcl::PointCloud<pcl::PointXYZI> &cloud) 
{	
	FILE *file = fopen(bin_file.c_str(), "rb");
	std::ifstream label_input(label_file, std::ios::binary);
	if (!file || !label_input.is_open())
	{
		throw invalid_argument("Could not open the .bin or .label file!");
	}
	else 
	{
		std::vector<float> buffer(1000000);
		size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
		fclose(file);
		cloud.points.resize(num_points);
		label_input.seekg(0, std::ios::beg);
		std::vector<uint32_t> labels(num_points);
		label_input.read((char*)&labels[0], num_points * sizeof(uint32_t));

		for (int i = 0; i < num_points; i++) 
		{
			auto &pt = cloud.at(i);
			pt.x = buffer[i * 4];
			pt.y = buffer[i * 4 + 1];
			pt.z = buffer[i * 4 + 2];
			std::uint32_t label = labels[i];
			pt.intensity = static_cast<float>(label);
		}
	}
}

// template<typename T>
// inline float pointDistance(T p) 
// {
// 	return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
// }

// template<typename T>
// inline float pointDistance(T p1, T p2) 
// {
// 	return std::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z));
// }

template <typename T>
class PubGridMap
{
public:
	PubGridMap();
	PubGridMap(ros::NodeHandle& nh_, string topic_);
	~PubGridMap();

	void publish(int index_, T msg_);

private:
	ros::NodeHandle nh;
	string topic;
	std::vector<std::shared_ptr<ros::Publisher>> pub_grid_map;
};

template<typename T>
PubGridMap<T>::PubGridMap() {}

template<typename T>
PubGridMap<T>::PubGridMap(ros::NodeHandle& nh_, string topic_) 
{
	nh = nh_;
	topic = topic_; 
	pub_grid_map.resize(1e3);
}

template<typename T>
PubGridMap<T>::~PubGridMap() {}

template<typename T>
void PubGridMap<T>::publish(int index_, T msg_) 
{
	if (pub_grid_map[index_] == nullptr) 
	{
		pub_grid_map[index_] = std::make_shared<ros::Publisher>(nh.advertise<T>(
		topic+std::to_string(index_), 1));
	}
	pub_grid_map[index_]->publish(msg_);
	// printf("pub %d-th grid map\n", index_);
}