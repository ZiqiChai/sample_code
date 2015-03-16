#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>


#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/filters/passthrough.h>

using namespace pcl::console;

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
  if (argc < 5)
  {
    printf ("No target PCD file and scale given!\n");
    printf ("Usage: scale_pcd <file.pcd> <output.pcd> <direction> <min_depth> <max_depth>\n");
    return (-1);
  }

   std::string output = argv[2];
  double min_depth = atof(argv[4]); 
  double max_depth = atof(argv[5]);
  std::string direction = argv[3];
  if(direction.empty())
  {
	
  }
  print_highlight (stderr, "Loading "); print_value (stderr, "%s\n ", argv[1]);
  
  pcl::console::TicToc tt;
  tt.tic ();
 
  // Load the target cloud PCD file
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::io::loadPCDFile (argv[1], *cloud);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = convert(cloud);

 print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms \n"); 

 print_highlight (stderr, "Running PassThrought filter in %s direction with min_depth ", direction.c_str()); 
 print_value (stderr, "%6.6f", min_depth);
 print_highlight (stderr, " and max_depth ");
 print_value (stderr, "%6.6f\n", max_depth);

 pcl::PassThrough<pcl::PointXYZRGB> pass;
 pass.setInputCloud (rgb_cloud);
 pass.setFilterFieldName (direction);
 pass.setFilterLimits (min_depth, max_depth);
 pass.filter (*rgb_cloud);

 print_highlight (stderr, "Saving %s\n ", output.c_str()); 
  // Save the aligned template for visualization
  pcl::io::savePCDFileASCII(output, *rgb_cloud);

  return (0);
}
