#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include "ReconstructPointCloud.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

using namespace pcl::console;


void showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             pose_estimation_app - Usage Guide              		     *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <input.ply> <output.ply> [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:					Show this help." << std::endl;
  std::cout << "     -k:		                       Show used keypoints." << std::endl;
  std::cout << "     -c:		                       Show used correspondences." << std::endl;
  std::cout << "     --vis:					Show visualization" << std::endl;
  std::cout << "     --plane:         	      			Remove dominante plane (default: true)" << std::endl;
  std::cout << "     --plane_thres:				Plane removal distance threshold  (default 0.005)" << std::endl;  
  std::cout << "     --algorithm (RANSAC|Prereject|CG|PPF): 	Algorithm used (default: Prereject)." << std::endl;
  std::cout << "     --icp:					Use ICP fine registration." << std::endl;
  std::cout << "     --leaf_size:				Voxel_grid leaf_size (default 0.006)" << std::endl;
  std::cout << "     --save:					Save aligned cloud (default: false)" << std::endl; 
  std::cout << "     --gt:					Load ground truth pose from file (default: "")" << std::endl; 
  std::cout << "     --gt_ignore_z_rot:			Ignore rotation around z-axis in the error computation(default: 0)" << std::endl; 
  std::cout << "     --gt_ignore_y_rot:			Ignore rotation around y-axis in the error computation(default: 0)" << std::endl; 
  std::cout << "     --gt_ignore_x_rot:			Ignore rotation around x-axis in the error computation(default: 0)" << std::endl; 
  std::cout << "     --gt_flip_around_x:			Flip model around x-axis using a correction Matrix provided in file x(default:"")" << std::endl; 
  std::cout << "***************************************************************************" << std::endl << std::endl;
   
}

bool
loadCloud_ply (const std::string &filename, PointCloudNT &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (pcl::io::loadPLYFile(filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
saveCloud_ply (const std::string &filename, const PointCloudNT &output)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());
  pcl::io::savePLYFile (filename, output);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}

// Scale a point cloud
int main (int argc, char **argv)
{
  if (argc < 2)
  {
    printf ("No target PCD file and scale given!\n");
    printf ("Usage: %s <file.pcd> <scale>\n",argv);
    return (-1);
  }

 
 
  print_highlight (stderr, "Loading "); print_value (stderr, "%s\n ", argv[1]); 
  PointCloudNT::Ptr cloud_in(new PointCloudNT()); 
  if(!loadCloud_ply(argv[1],*cloud_in));
     return 0;
  
  
 dti::surface::ReconstructPointCloud rec;
 
  

 
  

  return (0);
}
