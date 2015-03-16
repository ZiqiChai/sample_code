#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/filters/statistical_outlier_removal.h>

using namespace pcl::console;

void statisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud, int mean)
{
	 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud (src_cloud);
	sor.setMeanK (mean);
	sor.setStddevMulThresh (1.0);

	sor.filter (*cloud_filtered);

  	pcl::copyPointCloud(*cloud_filtered, *target_cloud);

}




pcl::PointCloud<pcl::PointXYZRGB>::Ptr convert(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr in) 
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>);
  out->points.resize(in->size());

  for (size_t i = 0; i < in->size(); ++i) {
    out->points[i].x = in->points[i].x;
    out->points[i].y = in->points[i].y;
    out->points[i].z = in->points[i].z;
    out->points[i].r = in->points[i].r;
    out->points[i].g = in->points[i].g;
    out->points[i].b = in->points[i].b;
  }
  return out;
}


// Scale a point cloud
int main (int argc, char **argv)
{
  if (argc < 3)
  {
    printf ("No target PCD file and scale given!\n");
    printf ("Usage: %s <file.pcd> <output.pcd> <mean>\n", argv[0]);
    return (-1);
  }

   std::string output = argv[2];
  double mean = atof(argv[3]); 
 
  print_highlight (stderr, "Loading "); print_value (stderr, "%s\n ", argv[1]);
  
  pcl::console::TicToc tt;
  tt.tic ();
 
  // Load the target cloud PCD file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile (argv[1], *cloud);

//pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = convert(cloud);

 print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms \n"); 

 print_highlight (stderr, "Running statistical Outlier Removal with mean =  "); 
 print_value (stderr, "%6.6f", mean);
 
 statisticalOutlierRemoval(cloud, cloud, mean);

 print_highlight (stderr, "Saving %s\n ", output.c_str()); 
  // Save the aligned template for visualization
  pcl::io::savePCDFileASCII(output, *cloud);

  return (0);
}
