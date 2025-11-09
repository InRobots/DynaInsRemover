#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <nav_msgs/OccupancyGrid.h>

template <typename PointT>
class InsGridMap
{
public:
    InsGridMap();
    ~InsGridMap();

	void setMapSize(const int num_);
    void setInputCloud(const pcl::PointCloud<PointT> &cloud);
	float update();
	void setProDynamic(float pro_);
	void setProStatic(float pro_);
	void setClampingThresMax(float pro_);
	void setClampingThresMin(float pro_);
	nav_msgs::OccupancyGrid getGridMap();

private:
	float pointDistance(PointT p);
	int getGridIndex(PointT &p);
	float logistic(float log_odd_);
	float logit(float pro_);
	float updateGridData(int index_, float ratio_);
	
private:
    int num;
	float grid_res;
	float lmax, ld, ls, lmin;

	pcl::PointCloud<PointT> point_cloud;
	std::vector<pcl::PointCloud<PointT>> ins_clouds;
	pcl::PointCloud<PointT> cluster_cloud;

	pcl::PointXYZI max_point, min_point;
	Eigen::Vector3f ins_box;

    float min_height =  999.9;
	float max_height = -999.9;

	float delta_height = 0.0;
	float lower_height = 0.0;
	float upper_height = 0.0;

	int width, height, grid_size; // x, y

	std::set<int> grid_index;
	std::unordered_map<int, float> grid_data;
	std::vector<float> log_odds;
	nav_msgs::OccupancyGrid grid_map;
};

template<typename PointT>
InsGridMap<PointT>::InsGridMap() {}

template<typename PointT>
InsGridMap<PointT>::~InsGridMap() {}

template<typename PointT> inline 
float InsGridMap<PointT>::pointDistance(PointT p) 
{
	return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

template<typename PointT> inline
float InsGridMap<PointT>::logit(float pro_) 
{
	return std::log(pro_/(1-pro_));
}

template<typename PointT> inline
float InsGridMap<PointT>::logistic(float log_odd_) 
{
	return 1/(1+std::exp(-log_odd_));
}

template<typename PointT> inline 
int InsGridMap<PointT>::getGridIndex(PointT &p) 
{
	int y = std::ceil((p.x - min_point.x) / grid_res) - 1;
	int x = std::ceil((p.y - min_point.y) / grid_res) - 1;

	if (x < 0)
		x = 0;
	if (y < 0)
		y = 0;

	return x * width + y;
}

template<typename PointT> inline
void InsGridMap<PointT>::setInputCloud(const pcl::PointCloud<PointT> &cloud) 
{
	printf("****************************************************\n");
	pcl::getMinMax3D(cloud, min_point, max_point);

	min_height = min_point.z;
	max_height = max_point.z;
	
	delta_height = max_height - min_height;

	if (1.0 < delta_height) 
	{
		lower_height = min_height + 0.2f * delta_height;
		upper_height = std::max(3.0f, min_height + 0.8f * delta_height);
	}
	else 
	{
		lower_height = min_height;
		upper_height = std::max(3.0f, max_height);
	}

	// separating point cloud
	for (PointT p : cloud.points) 
	{
		int index = p.intensity;
		if (lower_height <= p.z && p.z <= upper_height) 
		{
			ins_clouds[index].points.push_back(p);
			point_cloud.points.push_back(p);
		}
	}
	printf("map cloud num(%ld, %ld)\n", point_cloud.size(), cloud.size());

	pcl::getMinMax3D(point_cloud, min_point, max_point);
	ins_box = max_point.getVector3fMap() - min_point.getVector3fMap();

	// grid_res = std::max(0.1 * std::ceil(std::max(pointDistance(min_point), pointDistance(max_point)) / 10.0), 0.1);
	grid_res = std::min(std::max(0.1 * std::ceil(std::min(ins_box.x(), ins_box.y())/1.0), 0.1), 0.5);
	printf("adapt_res: %f\n", grid_res);

	width  = std::ceil((max_point.x - min_point.x) / grid_res);
	height = std::ceil((max_point.y - min_point.y) / grid_res);
	grid_size = width * height;

	for (PointT p : point_cloud.points) 
	{
		int index = getGridIndex(p);
		grid_index.insert(index);
		std::unordered_map<int, float>::iterator iter = grid_data.find(index);
		if (iter == grid_data.end()) 
		{
			grid_data[index] = 0;
		}
	}

	// for (int i = 0; i < num; ++i) 
	// {
	// 	printf("ins num(%d, %ld)\n", i, ins_clouds[i].size());
	// }

	printf("width: %d, height: %d\n", width, height);
}

template<typename PointT> inline
float InsGridMap<PointT>::update() 
{
	float map_log_odd = 0;

	std::set<int> map_index;
	std::unordered_map<int, float> map_data;

	for (PointT p : point_cloud.points) 
	{
		int index = getGridIndex(p);
		map_index.insert(index);
		std::unordered_map<int, float>::iterator iter = map_data.find(index);
		float value = p.z - min_height;

		if (iter == map_data.end())
		{
			map_data[index] = value;
			// printf("insert map grid: (%d, %f)\n", index, map_data[index]);
		}
		else
		{
			map_data[index] = std::max(map_data[index], value);
			// printf("update map grid: (%d, %f)\n", index, map_data[index]);
		}
	}

	for (int i = 0; i < num; ++i) 
	{
		// printf("iter: %d\n", i);
		log_odds[i] = 0;

		std::set<int> ins_index;
		std::unordered_map<int, float> ins_data;
		pcl::PointCloud<PointT>& ins_cloud = ins_clouds[i];

		if (ins_cloud.size() != 0) 
		{
			for (PointT p : ins_cloud.points) 
			{
				int index = getGridIndex(p);
				ins_index.insert(index);
				std::unordered_map<int, float>::iterator iter = ins_data.find(index);
				float value = p.z - min_height;

				if (iter == ins_data.end())
				{
					ins_data[index] = value;
					// printf("insert ins grid: (%d, %f)\n", index, ins_data[index]);
				}
				else
				{
					ins_data[index] = std::max(ins_data[index], value);
					// printf("update ins grid: (%d, %f)\n", index, ins_data[index]);
				}
			}

			for (std::set<int>::iterator iter = map_index.begin(); iter != map_index.end(); ++iter) 
			{
				int index = *iter;
				float mh = 0;
				std::unordered_map<int, float>::iterator mi = map_data.find(index);
				if (mi != map_data.end()) 
				{
					mh = mi->second;
				}

				float ih = 0;
				std::unordered_map<int, float>::iterator ii = ins_data.find(index);
				if (ii != ins_data.end()) 
				{
					ih = ii->second;
				}

				float ratio = 0;
				if (mh != 0 || ih != 0)
				{
					ratio = std::min(ih, mh) / std::max(ih, mh);
				}

				log_odds[i] += updateGridData(index, ratio);
				// log_odds[i] = std::max(std::min(log_odds[i], lmax), lmin);

				// printf("ind_map_ins_rat_odd(%d, %f, %f, %f, %f)\n", index, mh, ih, ratio, log_odds[i]);

				// printf("index: %d, map: %f, ins: %f, rat:%f, log_odd: %f)\n", index, mh, ih, ratio, log_odds[i]);

			}
		}
		
		printf("ins num(%d, %ld)\n", i, ins_clouds[i].size());
		printf("log-odds(%d, %f, %f, %f, %f)\n", i, log_odds[i], logistic(log_odds[i]), log_odds[i]/map_index.size(), logistic(log_odds[i]/map_index.size()));

	}

	for (float log_odd : log_odds) 
	{
		if (0.5 < logistic(log_odd/map_index.size())) 
		{
			map_log_odd += ld;
		}
		else 
		{
			map_log_odd += 2.0 * ls;
		}
	}

	return logistic(map_log_odd);
}

template<typename PointT> inline
float InsGridMap<PointT>::updateGridData(int index_, float ratio_) 
{
	float log_odd;
	if (0 == ratio_) 
	{
		log_odd = 2.0 * ld;

	}
	else if (0.8 < ratio_) 
	{
		log_odd = ls;
	}
	else 
	{
		log_odd = ld;
	}

	grid_data[index_] += log_odd;
	// printf("log_odd(%d, %f)\n",index_, log_odd);

	return log_odd;
}

template<typename PointT> inline
void InsGridMap<PointT>::setMapSize(const int num_) 
{
	num = num_;
	ins_clouds.resize(num);
	cluster_cloud.resize(num);
	log_odds.resize(num);
}

template<typename PointT> inline
void InsGridMap<PointT>::setProDynamic(float pro_) 
{
	ld = logit(pro_);
	// printf("log dynamic(%f, %f)\n", pro_, ld);
}

template<typename PointT> inline
void InsGridMap<PointT>::setProStatic(float pro_) 
{
	ls = logit(pro_);
	// printf("log Static(%f, %f)\n", pro_, ls);
}

template<typename PointT> inline
void InsGridMap<PointT>::setClampingThresMax(float pro_) 
{
	lmax = logit(pro_);
	// printf("logit max(%f, %f)\n", pro_, lmax);
}

template<typename PointT> inline
void InsGridMap<PointT>::setClampingThresMin(float pro_) 
{
	lmin = logit(pro_);
	// printf("log min(%f, %f)\n", pro_, lmin);
}

template<typename PointT> inline
nav_msgs::OccupancyGrid InsGridMap<PointT>::getGridMap() 
{
	// for (PointT p : point_cloud.points) 
	// {
	// 	int index = getGridIndex(p);
	// 	grid_index.insert(index);
	// 	std::unordered_map<int, float>::iterator iter = grid_data.find(index);
	// 	if (iter == grid_data.end()) 
	// 	{
	// 		grid_data[index] = 100;
	// 	}	
	// }

	grid_map.info.resolution = grid_res;
    grid_map.info.origin.position.x = min_point.x;
    grid_map.info.origin.position.y = min_point.y;
    grid_map.info.origin.position.z = min_point.z;
    grid_map.info.origin.orientation.x = 0.0;
    grid_map.info.origin.orientation.y = 0.0;
    grid_map.info.origin.orientation.z = 0.0;
    grid_map.info.origin.orientation.w = 1.0;

    grid_map.info.width  = width;
    grid_map.info.height = height;
    grid_map.data.resize(grid_size);

	for(int i = 0; i < grid_size; i++)
    {
		std::unordered_map<int, float>::iterator iter = grid_data.find(i);
		if (iter != grid_data.end()) 
		{
			grid_map.data[i] = 100;
		}
		else 
		{
			grid_map.data[i] = -1;
		}
		// grid_map.data[i] = grid_data[i];
    }

	return grid_map;
}