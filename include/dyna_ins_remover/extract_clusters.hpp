#ifndef CURVED_VOXEL_HPP
#define CURVED_VOXEL_HPP

#include <limits>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <boost/thread/thread.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#define PI 180.0/M_PI

struct RAP 
{
	float range = 0;
	float azimuth = 0;
	float polar = 0; 
};

struct VoxelIndex
{
   	int range_index = -1;
	int azimuth_index = -1;
	int polar_index = -1;
   	int voxel_index = -1;
};

struct Voxel
{
	bool valid = false;
	bool used = false;
	VoxelIndex index;
   	std::vector<int> point_indices;
};

template <typename PointT>
class CurvedVoxelClusterExtraction
{
public:
	CurvedVoxelClusterExtraction();
	~CurvedVoxelClusterExtraction();

	void setVoxelResolution(float range, float azimuth, float polar);
	void setInputCloud(const pcl::PointCloud<PointT> &cloud);
	int findNeighborVoxels(int range_index, int azimuth_index, int polar_index, int num, std::vector<Voxel>& voxels);

	std::vector<Voxel> getVoxels();
	std::vector<VoxelIndex> getVoxelIndices();
	std::unordered_map<int, Voxel> getHashVoxels();
	std::vector<std::vector<Voxel>> getClusters();

public:

	int length = 0;
	int width  = 0;
	int height = 0;

private:
	float calculateAzimuth(float x, float y);

private:
	float voxel_range   = 0.5;
	float voxel_azimuth = 0.5;
	float voxel_polar   = 0.5;

	float min_range = std::numeric_limits<float>::max();
	float max_range = std::numeric_limits<float>::min();

	float min_polar = std::numeric_limits<float>::max();
	float max_polar = std::numeric_limits<float>::min();

	std::vector<RAP> raps;
	std::vector<Voxel> voxels;
	std::vector<VoxelIndex> voxel_indices;
	std::unordered_map<int, Voxel> hash_voxels;
	std::vector<std::vector<Voxel>> clusters;

};

template<typename PointT>
CurvedVoxelClusterExtraction<PointT>::CurvedVoxelClusterExtraction() {}

template<typename PointT>
CurvedVoxelClusterExtraction<PointT>::~CurvedVoxelClusterExtraction() {}

template<typename PointT> inline
std::vector<Voxel> CurvedVoxelClusterExtraction<PointT>::getVoxels() {return voxels;}

template<typename PointT> inline
std::vector<VoxelIndex> CurvedVoxelClusterExtraction<PointT>::getVoxelIndices() {return voxel_indices;}

template<typename PointT> inline
std::unordered_map<int, Voxel> CurvedVoxelClusterExtraction<PointT>::getHashVoxels() {return hash_voxels;}

template<typename PointT> inline
void CurvedVoxelClusterExtraction<PointT>::setVoxelResolution(float range, float azimuth, float polar) 
{
	voxel_range = range;
	voxel_azimuth = azimuth;
	voxel_polar = polar;
}

template<typename PointT> inline
float CurvedVoxelClusterExtraction<PointT>::calculateAzimuth(float x, float y)
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
void CurvedVoxelClusterExtraction<PointT>::setInputCloud(const pcl::PointCloud<PointT> &cloud) 
{

	int size = cloud.size();
	raps.resize(size);

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < size; ++i) 
	{
		RAP rap;
		auto& point = cloud.points[i];
		rap.range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
		rap.azimuth = calculateAzimuth(point.x, point.y);
		rap.polar = std::atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * PI;

		if(rap.range < min_range)
		{
			min_range = rap.range;
		}
		if(rap.range > max_range)
		{
			max_range = rap.range;
		}

		if(rap.polar < min_polar)
		{
			min_polar = rap.polar;
		}
		if(rap.polar > max_polar)
		{
			max_polar = rap.polar;
		}

		raps[i] = rap;
	}

	length = std::ceil((max_range - min_range) / voxel_range);
	width  = std::ceil(360.0 / voxel_azimuth);
	height = std::ceil(((max_polar - min_polar)) / voxel_polar);

	int count = 0;
	for (int i = 0; i < size; ++i)
	{
		auto& rap = raps[i];
		int range_index = std::ceil((rap.range - min_range) / voxel_range) - 1;
		int azimuth_index = std::ceil(rap.azimuth / voxel_azimuth) - 1;
		int polar_index = std::ceil((rap.polar - min_polar) / voxel_polar) - 1;
		
		if (range_index < 0)
			range_index = 0;

		if (azimuth_index < 0)
			azimuth_index = 0;
		
		if (polar_index < 0)
			polar_index = 0;
			
		int voxel_index = range_index * width + azimuth_index + polar_index * length * width;

		std::unordered_map<int, Voxel>::iterator iter = hash_voxels.find(voxel_index);
		if (iter != hash_voxels.end())
		{
			iter->second.point_indices.push_back(i);
			voxels[count - 1].point_indices.push_back(i);
		}
		else
		{
			VoxelIndex index;
			index.range_index = range_index;
			index.azimuth_index = azimuth_index;
			index.polar_index = polar_index;
			index.voxel_index = count;

			Voxel voxel;
			voxel.valid =true;
			voxel.point_indices.push_back(i);
			voxel.index = index;

			hash_voxels.insert(std::make_pair(voxel_index, voxel));
			voxel_indices.push_back(index);
			voxels.push_back(voxel);
			++count;
		}
	}
}

template<typename PointT> inline
int CurvedVoxelClusterExtraction<PointT>::findNeighborVoxels(int range_index, int azimuth_index, int polar_index, int num, std::vector<Voxel>& voxels) 
{
	if (num == 0) 
		return 0;
	
	else if (num == 1) 
	{
		int voxel_index = range_index * width + azimuth_index + polar_index * length * width;
		std::unordered_map<int, Voxel>::iterator iter = hash_voxels.find(voxel_index);
		if (iter != hash_voxels.end())
		{
			voxels.push_back(iter->second);	
		}
	}

	else if (num == 3)
	{
		voxels.clear();
		int voxel_index = range_index * width + azimuth_index + polar_index * length * width;
		std::unordered_map<int, Voxel>::iterator iter = hash_voxels.find(voxel_index);
		if (iter != hash_voxels.end())
			voxels.push_back(iter->second);

		for (int z = polar_index - 1; z < polar_index + 2; ++z)
		{

			if (height <= z || z < 0)
				continue;
			
			for (int y = azimuth_index -1; y < azimuth_index + 2; ++y)
			{
				
				int py = y;
				if (y < 0)
					py = width - 1;
				if (width <= y)
					py = 0;

				for (int x = range_index - 1; x < range_index + 2; ++x) 
				{
					if (length <= x || x < 0)
						continue;
					
					if (x == range_index && y == azimuth_index && z == polar_index)
						continue;
						
					int voxel_index = x * width + py + z * length * width;

					std::unordered_map<int, Voxel>::iterator iter = hash_voxels.find(voxel_index);

					if (iter != hash_voxels.end())
					{
						voxels.push_back(iter->second);
					}
				}
			}
		}
	}
	return (int)voxels.size();
}

template<typename PointT> inline
std::vector<std::vector<Voxel>> CurvedVoxelClusterExtraction<PointT>::getClusters()
{
	int size = voxels.size();
	std::vector<bool> voxel_state = std::vector<bool>(size, false);

	for (int i = 0; i < size; ++i) 
	{
		if (!voxel_state[i]) 
		{

			std::vector<Voxel> this_cluster;
			int cluster_size = 1;
			int index = 0;

			while (0 < cluster_size) 
			{
				--cluster_size;
				Voxel voxel;
				if (index == 0) 
				{
					voxel = voxels[i];
				}
				else
				{
					if (this_cluster.size() <= index) 
						break;
						
					voxel = this_cluster[index];
				}

				index++;

				std::vector<Voxel> neighbor_voxels;
				findNeighborVoxels(voxel.index.range_index, voxel.index.azimuth_index, voxel.index.polar_index, 3, neighbor_voxels);

				for (auto neighbor_voxel : neighbor_voxels)
				{
					if (!voxel_state[neighbor_voxel.index.voxel_index])
					{
						voxel_state[neighbor_voxel.index.voxel_index] = true;
						this_cluster.push_back(neighbor_voxel);
						++cluster_size;
					}
				}
			}
			clusters.push_back(this_cluster);
		}
	}

	return clusters;
}

#endif