

#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_types_conversion.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

double ComputeCloudResolution(const pcl::PCLPointCloud2::ConstPtr &cloud){
 
  PointCloud<PointXYZ>::Ptr xyz_source (new PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*cloud, *xyz_source);
  
   pcl::search::KdTree<pcl::PointXYZ> s;
    const int k = 5;
    std::vector<std::vector<int> > idx;
    std::vector<std::vector<float> > distsq;

    s.setInputCloud(xyz_source);
    s.nearestKSearch(*xyz_source, std::vector<int>(), 5, idx, distsq);
    double res_src = 0.0f;
    for(size_t i = 0; i < xyz_source->size(); ++i) {
        double resi = 0.0f;
        for(int j = 1; j < k; ++j)
            resi += sqrtf(distsq[i][j]);
        resi /= double(k - 1);
        res_src += resi;
    }
    res_src /= double(xyz_source->size());
  
    return res_src;
}

bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", cloud.width * cloud.height);  	print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}


/* ---[ */
int
main (int argc, char** argv)
{
//  print_info ("Compute the differences between two point clouds and visualizing them as an output intensity cloud. For more information, use: %s -h\n", argv[0]);

  if (argc < 2)
  {
     print_error ("Syntax is: %s source.pcd \n", argv[0]);
    return (-1);
  }

  // Load the first file
  pcl::PCLPointCloud2::Ptr cloud_source (new pcl::PCLPointCloud2 ());
  if (!loadCloud (argv[1], *cloud_source))
    return (-1);
   print_highlight("Source cloud loaded with an average point resolution = %f mm\n", ComputeCloudResolution(cloud_source)*1000);
 
 return 0;
}
