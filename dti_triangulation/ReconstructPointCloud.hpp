/*
 * ReconstructPointCloud.h
 *
 *  Created on: Aug 13, 2013
 *      Author: thomas
 */

#ifndef RECONSTRUCTPOINTCLOUD_H_
#define RECONSTRUCTPOINTCLOUD_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> CloudNT;


namespace dti{
namespace surface {

class ReconstructPointCloud {
public:
	ReconstructPointCloud();
	virtual ~ReconstructPointCloud();

	void MLSApproximation(CloudT::Ptr &cloud, CloudT::Ptr &target);
	bool BilateralUpsampling(CloudT::Ptr cloud, CloudT::Ptr output,int window_size, double sigma_color, double sigma_depth);

	void poisson(const CloudNT::Ptr &cloud, pcl::PolygonMesh &output,int depth, int solver_divide, int iso_divide, float point_weight);
	pcl::PolygonMesh GreedyProjectionTriangulation(CloudNT::Ptr cloud_with_normals, double SearchRadius);
	pcl::PolygonMesh MarchingCubes(CloudNT::Ptr cloud_with_normals, double leafSize = 0.5, double isoLevel = 0.5);
	pcl::PolygonMesh GridProjection(CloudNT::Ptr cloud_with_normals);
	pcl::PolygonMesh OrganizedFastMaesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr ConvexHull(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr loadPCDfile(std::string file_path);
	void saveToObj(const std::string file, pcl::PolygonMesh mesh);
	void saveToVTK(const std::string file, pcl::PolygonMesh mesh);
};

}
} /* namespace object_modeller_gui */
#endif /* RECONSTRUCTPOINTCLOUD_H_ */
