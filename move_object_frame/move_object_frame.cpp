#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/pca.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/PCLPointCloud2.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s input.pcd output.pcd  <options>\n", argv[0]);
 // print_info ("  where options are:\n");
}

bool moveObjectFrame(pcl::PCLPointCloud2::Ptr &src_cloud, pcl::PCLPointCloud2::Ptr &tar_cloud){
  
	CloudT::Ptr source (new CloudT ());
	fromPCLPointCloud2 (*src_cloud, *source);
 	
	Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*source,centroid);
	
	//  pcl::transformPointCloud(*src_cloud,*temp,mat);
		 
	pcl::PCA<PointT> _pca; 
	PointT projected; 
	PointT reconstructed;
	CloudT cloudi = *source;
	CloudT finalCloud;
		 
	try{
	     //Do PCA for each point to preserve color information
	     //Add point cloud to force PCL to init_compute else a exception is thrown!!!HACK
	     _pca.setInputCloud(source);
	     int i;
	 //    #pragma omp parallel for
	     for(i = 0; i < (int)source->size(); i++)     {
	       _pca.project(cloudi[i],projected);
	       Eigen::Matrix3f eigen = _pca.getEigenVectors();
	       
	       // flip axis to satisfy right-handed system
              if (eigen.col(0).cross(eigen.col(1)).dot(eigen.col(2)) < 0) {
		        projected.z = -projected.z; //Avoid flipping the model
              }
              
	       _pca.reconstruct (projected, reconstructed);

	       if(pcl::getFieldIndex(*src_cloud,"rgba") >= 0){
		 //assign colors
		 projected.r = cloudi[i].r;
		 projected.g = cloudi[i].g;
		 projected.b = cloudi[i].b;
	       }
	       //add point to cloud
	       finalCloud.push_back(projected);
	       
	    }
	    
	}catch(pcl::InitFailedException &e){
	  PCL_ERROR(e.what());
	  
	}

	 toPCLPointCloud2 (finalCloud, *tar_cloud);
	//pcl::copyPointCloud(finalCloud,*tar_cloud);
}

bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &output)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());
  pcl::io::savePCDFile (filename, output);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}

int
main (int argc, char** argv)
{

  if (argc < 2)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd files
  std::vector<int> p_file_indices;
  p_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  if (p_file_indices.size () != 2)
  {
    print_error ("Need one input PCD file and one output PCD file to continue.\n");
    return (-1);
  }

  // Command line parsing

  // Load the first file
  pcl::PCLPointCloud2::Ptr cloud_source (new pcl::PCLPointCloud2 ());
  if (!loadCloud (argv[p_file_indices[0]], *cloud_source))
    return (-1);
  
  pcl::PCLPointCloud2::Ptr cloud_target (new pcl::PCLPointCloud2 ());
  moveObjectFrame(cloud_source, cloud_target);
  saveCloud(argv[p_file_indices[1]], *cloud_target);

return 0;
}

