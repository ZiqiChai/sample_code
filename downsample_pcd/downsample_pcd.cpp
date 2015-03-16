#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/distances.h>
#include <pcl/search/kdtree.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

//Downsampling using the voxel grid approach
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointXYZRGBNormal PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace pcl::console;

double ComputeCloudResolution(const PointCloudT::Ptr &cloud)
{  
   pcl::search::KdTree<PointT> s;
    const int k = 5;
    std::vector<std::vector<int> > idx;
    std::vector<std::vector<float> > distsq;

    s.setInputCloud(cloud);
    s.nearestKSearch(*cloud, std::vector<int>(), 5, idx, distsq);
    double res_src = 0.0f;
    for(size_t i = 0; i < cloud->size(); ++i) {
        double resi = 0.0f;
        for(int j = 1; j < k; ++j)
            resi += sqrtf(distsq[i][j]);
        resi /= double(k - 1);
        res_src += resi;
    }
    res_src /= double(cloud->size());
  
    return res_src;
}


PointCloudT::Ptr DownSampleCloud(PointCloudT::Ptr cloud, double leaf_size)
{
	 std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
	       << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

	 PointCloudT::Ptr cloud_filtered (new PointCloudT);
	// Create the filtering object
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud (cloud);
	sor.setLeafSize (leaf_size,leaf_size,leaf_size);
	sor.filter (*cloud_filtered);

	 std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
	       << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

	return cloud_filtered;
}

// Scale a point cloud
int main (int argc, char **argv)
{
  if (argc < 2)
  {
    printf ("No target PCD file and downsample rate given!\n");
    printf ("Usage: downsample_pcd <file.pcd> <scale>\n");
    return (-1);
  }

 
  double downsample_rate = atof(argv[2]);
  print_highlight (stderr, "Loading "); print_value (stderr, "%s\n ", argv[1]);
  
  pcl::console::TicToc tt;
  tt.tic ();
 
  // Load the target cloud PCD file
  PointCloudT::Ptr cloud (new PointCloudT);
  pcl::io::loadPCDFile (argv[1], *cloud);

 print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms \n"); 

 print_highlight (stderr, "Downsampling point cloud with rate "); print_value (stderr, "%6.6f\n ", downsample_rate);

 PointCloudT::Ptr cl = DownSampleCloud(cloud,  downsample_rate);
 double res = ComputeCloudResolution(cl);
 print_highlight (stderr, "Cloud average point resolution = "); print_value (stderr, "%6.6f\n ", res);
  // Save the aligned template for visualization
 // pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
 // pcl::transformPointCloud (*best_template.getPointCloud (), transformed_cloud, best_alignment.final_transformation);
  pcl::io::savePCDFileASCII("output.pcd", *cl);

  return (0);
}
