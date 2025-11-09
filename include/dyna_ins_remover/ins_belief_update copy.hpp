#ifndef INS_BELIEF_UPDATE_HPP
#define INS_BELIEF_UPDATE_HPP

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PolygonStamped.h>

#define PI 180.0/M_PI

template <typename PointT>
class InstanceBeliefUpdate
{
public:
    InstanceBeliefUpdate();
    ~InstanceBeliefUpdate();

    void setResolution(float range, float azimuth);
    void setInputCloud(const std::vector<pcl::PointCloud<PointT>>& clouds);
	void setProDynamic(float pro);
	void setProStatic(float pro);
	float getInsBelief();
	jsk_recognition_msgs::PolygonArray getGridMap(const std_msgs::Header& header);

private:
	float calculateAzimuth(float x, float y);
	float logit(float pro);
	float logistic(float log_odd);
	int getGridIndex(PointT &p);
	void updateBelief();
	float updateGridData(int index, float ratio);

public:
	int length;
	int width;
    
private:
	int window_size;
	std::vector<pcl::PointCloud<PointT>> ins_clouds;
	std::vector<int> nums;

	pcl::PointCloud<PointT> map_cloud;

	bool cut = false;

	float min_range =  9999.9f;
	float max_range = -9999.9f;
    float delta_range = 0.0f;

    float min_azimuth =  9999.9f;
	float max_azimuth = -9999.9f;
	float delta_azimuth = 0.0f;

    float min_height =  9999.9f;
	float max_height = -9999.9f;
	float delta_height = 0.0f;

    float lower_height = 0.0f;
	float upper_height = 0.0f;

	float grid_range = 0.1f;
	float grid_azimuth = 0.1f;

	std::set<int> grid_index;
	std::unordered_map<int, float> map_data;
	std::unordered_map<int, float> grid_data;

	float log_odd = 0.0f;
	float ld = 0.0f, ls = 0.0f;

	std::vector<float> ins_log_odds;
};

template<typename PointT>
InstanceBeliefUpdate<PointT>::InstanceBeliefUpdate() {}

template<typename PointT>
InstanceBeliefUpdate<PointT>::~InstanceBeliefUpdate() {}

template<typename PointT> inline
void InstanceBeliefUpdate<PointT>::setResolution(float range, float azimuth) 
{
    grid_range = range;
    grid_azimuth = azimuth;
}

template<typename PointT>
void InstanceBeliefUpdate<PointT>::setInputCloud(const std::vector<pcl::PointCloud<PointT>>& clouds) 
{
	window_size = clouds.size();

	for (auto& ins_cloud : clouds) 
	{
		// printf("ins point num: %ld\n", ins_cloud.size());
		if(0 < ins_cloud.size()) 
		{
			for (auto& p : ins_cloud.points) 
			{
				float azimuth = calculateAzimuth(p.x, p.y);
		        float height  = p.z;

				if(azimuth < min_azimuth)
				{
					min_azimuth = azimuth;
				}
				if(azimuth > max_azimuth)
				{
					max_azimuth = azimuth;
				}

				if(height < min_height)
				{
					min_height = height;
				}
				if(height > max_height)
				{
					max_height = height;
				}
			}
		}
	}

	delta_azimuth = max_azimuth - min_azimuth;

	if (180.0f < delta_azimuth) 
	{
		cut = true;
		min_azimuth =  9999.9f;
		max_azimuth = -9999.9f;
	}
	else
		cut = false;

	delta_height  = max_height - min_height;
	if (1.0f < delta_height) 
	{
		lower_height = min_height + 0.2f * delta_height;
		upper_height = max_height;
	}
	else 
	{
		lower_height = min_height;
		upper_height = max_height;
	}

	ins_clouds.resize(window_size);

	for (int i = 0; i < window_size; ++i) 
	{
		auto& cloud = clouds[i];
		auto& ins_cloud = ins_clouds[i];

		for (auto p : cloud.points) 
		{
			float range   = std::sqrt(p.x * p.x + p.y * p.y);
			float azimuth = calculateAzimuth(p.x, p.y);
			if (cut == true && 180.0f < azimuth)
			{
				azimuth = azimuth - 360.0f;
			}
			float height  = p.z;

			if (lower_height <= height && height <= upper_height) 
			{
				map_cloud.points.push_back(p);
				ins_cloud.points.push_back(p);

				if(range < min_range)
				{
					min_range = range;
				}
				if(range > max_range)
				{
					max_range = range;
				}

				if(azimuth < min_azimuth)
				{
					min_azimuth = azimuth;
				}
				if(azimuth > max_azimuth)
				{
					max_azimuth = azimuth;
				}
			}
		}
		
		int num = ins_cloud.size();
		nums.push_back(num);
		ins_log_odds.push_back(0.0f);

		// printf("ins point num: %ld\n", ins_clouds[i].size());
	}
	
	delta_range   = max_range - min_range;
	delta_azimuth = max_azimuth - min_azimuth;

	length = std::ceil(delta_range/grid_range);
	width  = std::ceil(delta_azimuth/grid_azimuth);

	if (length == 0) 
	{
		length = 1;
	}

	if (width == 0) 
	{
		width = 1;
	}

	printf("map point num: %ld\n", map_cloud.size());
	printf("length: %d, width: %d\n", length, width);
	printf("grid index num: %ld\n", grid_index.size());
	printf("grid data  num: %ld\n", map_data.size());

}

template<typename PointT> inline
float InstanceBeliefUpdate<PointT>::getInsBelief() 
{
	updateBelief();
	return logistic(log_odd);
}

template<typename PointT> inline
float InstanceBeliefUpdate<PointT>::updateGridData(int index, float ratio) 
{
	float update_log_odd;
	if (0 == ratio) 
	{
		update_log_odd = 3.0 * ld;

	}
	else if (0.9 < ratio) 
	{
		update_log_odd = ls;
	}
	else 
	{
		update_log_odd = ld;
	}

	// grid_data[index_] += log_odd;
	// printf("log_odd(%d, %f)\n",index_, log_odd);

	return update_log_odd;
}

template<typename PointT> inline
void InstanceBeliefUpdate<PointT>::updateBelief() 
{

	for (auto& p : map_cloud.points) 
	{
		int index = getGridIndex(p);
		grid_index.insert(index);

		float value = p.z - min_height;
		std::unordered_map<int, float>::iterator iter = map_data.find(index);
		if (iter == map_data.end())
		{
			map_data[index] = value;
			grid_data[index] = 0.0f;
			// printf("insert map grid: (%d, %f)\n", index, map_data[index]);
		}
		else
		{
			map_data[index] = std::max(map_data[index], value);
			// printf("update map grid: (%d, %f)\n", index, map_data[index]);
		}
	}

	for (int i = 0; i < window_size; ++i) 
	{
		if (0 < nums[i]) 
		{
			// printf("%d-th update\n", i);
			auto& ins_cloud = ins_clouds[i];
			std::unordered_map<int, float> ins_data;

			for (auto& p : ins_cloud.points) 
			{
				int index = getGridIndex(p);
				float value = p.z - min_height;
				std::unordered_map<int, float>::iterator iter = ins_data.find(index);
				if (iter == ins_data.end())
				{
					ins_data[index] = value;
					// printf("insert map grid: (%d, %f)\n", index, map_data[index]);
				}
				else
				{
					ins_data[index] = std::max(ins_data[index], value);
					// printf("update map grid: (%d, %f)\n", index, map_data[index]);
				}
			}

			for (std::set<int>::iterator iter = grid_index.begin(); iter != grid_index.end(); ++iter) 
			{
				int index = *iter;
				float mv = 0;
				std::unordered_map<int, float>::iterator mi = map_data.find(index);
				if (mi != map_data.end()) 
				{
					mv = mi->second;
				}

				float iv = 0;
				std::unordered_map<int, float>::iterator ii = ins_data.find(index);
				if (ii != ins_data.end()) 
				{
					iv = ii->second;
				}

				float ratio = 0;
				if (mv != 0 || iv != 0)
				{
					ratio = std::min(iv, mv) / std::max(iv, mv);
				}

				float update_log_odd = updateGridData(index, ratio);
				ins_log_odds[i] += update_log_odd;

				// printf("ind_map_ins_rat_odd(%d, %f, %f, %f, %f)\n", index, mv, iv, ratio, update_log_odd);

			}
		}
	}


	for (int i = 0; i < window_size; ++i) 
	{
		float ins_log_odd = ins_log_odds[i];
		if (ins_log_odd != 0) 
		{
			printf("%d-th ins (%f, %f), mean(%f, %f)\n", i, ins_log_odds[i], logistic(ins_log_odds[i]), ins_log_odds[i]/(float)map_data.size(), logistic(ins_log_odds[i]/(float)map_data.size()));

			if (0.65 < logistic(ins_log_odd)) 
			{
				log_odd += ld;
			}
			else 
			{
				log_odd += ls;
		}
		}
		// if (0.5 < logistic(log_odd/map_index.size())) 
		// {
		// 	map_log_odd += ld;
		// }
		// else 
		// {
		// 	map_log_odd += 2.0 * ls;
		// }
	}
}


template<typename PointT> inline
jsk_recognition_msgs::PolygonArray InstanceBeliefUpdate<PointT>::getGridMap(const std_msgs::Header& header) 
{
    jsk_recognition_msgs::PolygonArray grid_map;
	grid_map.header = header;
	grid_map.polygons.resize(length*width);
	grid_map.likelihood.resize(length*width);

	for (int l = 0; l < length; ++l) 
	{
		for (int w = 0; w < width; ++w)
		{
			int index = l * width + w;
			geometry_msgs::PolygonStamped polygon;
			polygon.header = header;
			
			geometry_msgs::Point32  point;

			// RL
			float range = l * grid_range + min_range;
    		float angle = (w * grid_azimuth + min_azimuth) / 180.0f * M_PI;
			point.x = range * cos(angle);
    		point.y = range * sin(angle);
			point.z = lower_height;
    		polygon.polygon.points.push_back(point);

			// RU
			range = range + grid_range;
			point.x = range * cos(angle);
			point.y = range * sin(angle);
			point.z = lower_height;
			polygon.polygon.points.push_back(point);

			// RU -> LU
			for (int i = 1; i <= 3; ++i) 
			{
				angle = angle + (grid_azimuth / 3.0f) / 180.0f * M_PI;
				point.x = range * cos(angle);
				point.y = range * sin(angle);
				point.z = lower_height;
				polygon.polygon.points.push_back(point);
			}

			// LL
			range = range - grid_range;
			point.x = range * cos(angle);
			point.y = range * sin(angle);
			point.z = lower_height;
			polygon.polygon.points.push_back(point);

			// LL->RL
			for (int i = 1; i < 3; ++i) {
				angle = angle - (grid_azimuth / 3.0f) / 180.0f * M_PI;
				point.x = range * cos(angle);
				point.y = range * sin(angle);
				point.z = lower_height;
				polygon.polygon.points.push_back(point);
			}

			grid_map.polygons[index] = polygon;
			grid_map.likelihood[index] = 0.0;
		}
	}

	for (auto& p : map_cloud.points) 
	{
		int index = getGridIndex(p);
		grid_map.likelihood[index] = 1.0;

		// printf("cur(%d, %d)\n", l, w);
	}

	return grid_map;
}

template<typename PointT> inline 
int InstanceBeliefUpdate<PointT>::getGridIndex(PointT &p) 
{
	float range   = std::sqrt(p.x * p.x + p.y * p.y);
	float azimuth = calculateAzimuth(p.x, p.y);

	if (cut == true && 180.0f < azimuth) 
	{
		azimuth = azimuth -360.0f;
	}

	int l = std::ceil((range - min_range) / grid_range) - 1;
	int w = std::ceil((azimuth - min_azimuth) / grid_azimuth) - 1;

	if (l < 0)
		l = 0;

	if (w < 0)
		w = 0;
	
	// printf("pre(%d, %d)\n", l, w);
	return l * width + w;
}

template<typename PointT> inline
float InstanceBeliefUpdate<PointT>::calculateAzimuth(float x, float y)
{
	float angle = 0;
	if(x == 0 && y == 0)
	{
		angle = 0;
	}
	else if(0 <= y)
	{
		angle = std::atan2(y, x);
	}
	else if(y < 0)
	{
		angle = std::atan2(y, x) + 2 * M_PI;
	}
	return angle * PI;
}

template<typename PointT> inline
float InstanceBeliefUpdate<PointT>::logit(float pro)
{
	return std::log(pro/(1-pro));
}

template<typename PointT> inline
float InstanceBeliefUpdate<PointT>::logistic(float log_odd)
{
	return 1/(1+std::exp(-log_odd));
}

template<typename PointT> inline
void InstanceBeliefUpdate<PointT>::setProDynamic(float pro) 
{
	ld = logit(pro);
	printf("log dynamic(%f, %f)\n", pro, ld);
}

template<typename PointT> inline
void InstanceBeliefUpdate<PointT>::setProStatic(float pro) 
{
	ls = logit(pro);
	printf("log Static(%f, %f)\n", pro, ls);
}

#endif