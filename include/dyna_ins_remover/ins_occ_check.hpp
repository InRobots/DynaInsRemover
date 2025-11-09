#ifndef INS_OCC_CHECK_HPP
#define INS_OCC_CHECK_HPP

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#define PI 180.0/M_PI

template <typename PointT>
class InstanceOccupancyCheck
{
public:
    InstanceOccupancyCheck();
    ~InstanceOccupancyCheck();

    void setResolution(float range, float azimuth);
    void setInputCloud(const pcl::PointCloud<PointT> &source, const pcl::PointCloud<PointT> &target);
    void computeDescriptor();
    void computeSimilarity(float& s);

    Eigen::MatrixXf getHeadDesc();
    Eigen::MatrixXf getTailDesc();

private:
	float calculateAzimuth(float x, float y);

public:
	int length;
	int width;

    Eigen::MatrixXf source_mat;
    Eigen::MatrixXf target_mat;
    
private:

	float min_range =  9999.9;
	float max_range = -9999.9;
    
    float min_azimuth =  9999.9;
	float max_azimuth = -9999.9;

	float min_polar =  9999.9;
	float max_polar = -9999.9;

    float min_height =  9999.9;
	float max_height = -9999.9;
    float lower_height = 0.0;

	float grid_range = 0.1;
	float grid_azimuth = 0.1;

    std::vector<float> source_range;
    std::vector<float> target_range;

    std::vector<float> source_azimuth;
    std::vector<float> target_azimuth;

    std::vector<float> source_polar;
    std::vector<float> target_polar;

    std::vector<float> source_height;
    std::vector<float> target_height;

};

template<typename PointT>
InstanceOccupancyCheck<PointT>::InstanceOccupancyCheck() {}

template<typename PointT>
InstanceOccupancyCheck<PointT>::~InstanceOccupancyCheck() {}

template<typename PointT> inline
Eigen::MatrixXf InstanceOccupancyCheck<PointT>::getHeadDesc() { return target_mat; }

template<typename PointT> inline
Eigen::MatrixXf InstanceOccupancyCheck<PointT>::getTailDesc() { return source_mat; }

template<typename PointT> inline
void InstanceOccupancyCheck<PointT>::setResolution(float range, float azimuth) 
{
    grid_range = range;
    grid_azimuth = azimuth;
}

template<typename PointT> inline
void InstanceOccupancyCheck<PointT>::setInputCloud(const pcl::PointCloud<PointT> &source, const pcl::PointCloud<PointT> &target) 
{
    for (auto point : source.points) 
    {
        float range   = std::sqrt(point.x * point.x + point.y * point.y);
        float azimuth = calculateAzimuth(point.x, point.y);
        float polar   = std::atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * PI;
        float height  = point.z;

        source_range.push_back(range);
        source_azimuth.push_back(azimuth);
        source_polar.push_back(polar);
        source_height.push_back(height);

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

        if(polar < min_polar)
		{
			min_polar = polar;
		}
		if(polar > max_polar)
		{
			max_polar = polar;
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

    for (auto point : target.points) 
    {
        float range   = std::sqrt(point.x * point.x + point.y * point.y);
        float azimuth = calculateAzimuth(point.x, point.y);
        float polar   = std::atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * PI;
        float height = point.z;

        target_range.push_back(range);
        target_azimuth.push_back(azimuth);
        target_polar.push_back(polar);
        target_height.push_back(height);

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

        if(polar < min_polar)
		{
			min_polar = polar;
		}
		if(polar > max_polar)
		{
			max_polar = polar;
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

    if (1.0 < (max_height - min_height)) 
    {
        lower_height = min_height + 0.2f * (max_height - min_height);
    }
    else 
    {
        lower_height = min_height;
    }

    float delta_azimuth = max_azimuth - min_azimuth;
    if (180.0f < delta_azimuth) 
    {
        float tmp_min_azimuth =  9999.9;
	    float tmp_max_azimuth = -9999.9;

        for (float azimuth : target_azimuth) 
        {
            if (azimuth < 180.0f) 
            {
                if (azimuth > tmp_max_azimuth) 
                {
                    tmp_max_azimuth = azimuth;
                }
            }
            else 
            {
                if (azimuth < tmp_min_azimuth) 
                {
                    tmp_min_azimuth = azimuth;
                }
            }
        }

        for (float azimuth : source_azimuth) 
        {
            if (azimuth < 180.0f) 
            {
                if (azimuth > tmp_max_azimuth) 
                {
                    tmp_max_azimuth = azimuth;
                }
            }
            else 
            {
                if (azimuth < tmp_min_azimuth) 
                {
                    tmp_min_azimuth = azimuth;
                }
            }
        }

        // printf("large delta azimuth: %f, need to convert!\n", delta_azimuth);
        // printf("min_azimuth: %f, max_azimuth: %f, point num: %ld\n", tmp_min_azimuth, tmp_max_azimuth, target_height.size() + source_height.size());
    }

	length = std::ceil((max_range - min_range) / grid_range);
	width  = std::ceil((max_azimuth - min_azimuth) / grid_azimuth);

	if (length == 0) 
	{
		length = 1;
	}

	if (width == 0) 
	{
		width = 1;
	}

    // printf(" min_range: %f, max_range: %f, grid_range: %f,\n min_azimuth: %f, max_azimuth: %f, grid_azimuth: %f,\n min_height: %f, max_height: %f, delta_height: %f,\n length: %d, width: %d, height: %d\n", min_range, max_range, (max_range - min_range), min_azimuth, max_azimuth, (max_azimuth - min_azimuth), min_height, max_height, (max_height - min_height), length, width, height);
}

template<typename PointT> inline
void InstanceOccupancyCheck<PointT>::computeDescriptor() 
{
    source_mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();
    target_mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();

    source_mat.resize(length, width);
    target_mat.resize(length, width);

    source_mat.setZero();
    target_mat.setZero();

    int source_size = source_range.size();
    for (int i = 0; i < source_size; ++i) 
    {
        if (lower_height < source_height[i]) 
        {
            int l = std::ceil((source_range[i] - min_range) / grid_range) - 1;
            int w = std::ceil((source_azimuth[i] - min_azimuth) / grid_azimuth) - 1;
            if (l < 0)
                l = 0;
            if (w < 0)
                w = 0;
            source_mat(l, w) = std::max((source_height[i] - min_height), source_mat(l, w));
        }
    }

    int target_size = target_range.size();
    for (int i = 0; i < target_size; ++i) 
    {
        if (lower_height < target_height[i]) 
        {
            int l = std::ceil((target_range[i] - min_range) / grid_range) - 1;
            int w = std::ceil((target_azimuth[i] - min_azimuth) / grid_azimuth) - 1;
            if (l < 0)
                l = 0;
            if (w < 0)
                w = 0;
            target_mat(l, w) = std::max((target_height[i] - min_height), target_mat(l, w));
        }
    }
}

template<typename PointT> inline
void InstanceOccupancyCheck<PointT>::computeSimilarity(float& sim) 
{
    Eigen::VectorXf source_vector = Eigen::VectorXf::Map(&source_mat(0), source_mat.size());
    Eigen::VectorXf target_vector = Eigen::VectorXf::Map(&target_mat(0), target_mat.size());
    float source_norm = source_vector.norm();
    float target_norm = target_vector.norm();
    if (source_norm == 0 || target_norm == 0) 
    {
        sim = 0;
    }
    else 
    {
        float cos = source_vector.dot(target_vector)/(source_vector.norm() * target_vector.norm());
        sim = cos;
    }

}

template<typename PointT> inline
float InstanceOccupancyCheck<PointT>::calculateAzimuth(float x, float y)
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

#endif