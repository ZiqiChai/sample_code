#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>


#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>


using namespace pcl::console;


// Scale a point cloud
int main (int argc, char **argv)
{
  if (argc < 2)
  {
    printf ("No target PCD file and scale given!\n");
    printf ("Usage: scale_pcd <file.pcd> <scale>\n");
    return (-1);
  }

 
  double scale = atof(argv[2]);
  print_highlight (stderr, "Loading "); print_value (stderr, "%s\n ", argv[1]);
  
  pcl::console::TicToc tt;
  tt.tic ();
 
  // Load the target cloud PCD file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile (argv[1], *cloud);

 print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms \n"); 

 print_highlight (stderr, "Converting point cloud with scale "); print_value (stderr, "%6.6f\n ", scale);

 if(cloud->height != 1){
 for(int i = 0; i<= cloud->height-1; i++)
 {
	 for(int j = 0; j<= cloud->width-1; j++){
		pcl::PointXYZRGB p =  cloud->at(j,i);
	 }

 }
 }else{
	 for (size_t i = 0; i < cloud->points.size(); ++i){
		 cloud->points[i].x = cloud->points[i].x * scale;
		 cloud->points[i].y = cloud->points[i].y * scale;
		 cloud->points[i].z = cloud->points[i].z * scale;
		 cloud->points[i].r = cloud->points[i].r;
		 cloud->points[i].g = cloud->points[i].g;
		 cloud->points[i].b = cloud->points[i].b;
	        // print_info ("Cannot access a unorganized point cloud");
	 }
	 //print_info ("Cannot access a unorganized point cloud");
 }
  // Save the aligned template for visualization
 // pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
 // pcl::transformPointCloud (*best_template.getPointCloud (), transformed_cloud, best_alignment.final_transformation);
  pcl::io::savePCDFileASCII("output.pcd", *cloud);

  return (0);
}
