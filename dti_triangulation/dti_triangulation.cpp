#include <limits>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>


#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "ReconstructPointCloud.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

using namespace pcl::console;

float leaf_size_;
float greedy_search = 0.025;
int poisson_depth_ = 12;  
int poisson_solver_divide_ = 8; 
int poisson_iso_divide_ = 5; 
float poisson_point_weight_ = 1;
float marching_leafSize_ = 0.5;
float marching_isoLevel_ = 0.5;

bool show_vis_ = false;
bool use_greedy_ = false;
bool use_grid_ = false;
bool use_marchingcube_ = false;
bool use_poisson_ = false;

void showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             dti_triangulation - Usage Guide              		     *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <input.ply> <output.ply> [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:					Show this help." << std::endl;
  std::cout << "     -k:		                       Show used keypoints." << std::endl;
  std::cout << "     -c:		                       Show used correspondences." << std::endl;
  std::cout << "     --vis:					Show visualization" << std::endl;
  std::cout << "     --algorithm (Greedy|Grid|MarchingCubes|Poisson): 	Algorithm used (default: Prereject)." << std::endl;
  std::cout << "     --greedy_search:				Search radius for GreedyProjectionTriangulation algorithm (default: 0.025)." << std::endl;
  std::cout << "     --poisson_depth:				Poisson Octree depth (default: 12)" << std::endl;
  std::cout << "     --poisson_solver_divide:			Poisson Solver divide (default: 8)" << std::endl;
  std::cout << "     --poisson_iso_divide:			Poisson iso_divide (default: 5)" << std::endl;
  std::cout << "     --poisson_point_weight:			Poisson Point weight (default: 1)" << std::endl;
  std::cout << "     --marching_leaf_size:			Marching Cube leaf size (default: 0.5)" << std::endl;
  std::cout << "     --marching_isoLevel:			Marching Cube iso level (default: 0.5)" << std::endl;
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

void parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h")){
    showHelp (argv[0]);
    exit (0);
  }
  
   if (argc < 2)
   {
    printf ("No target .pcd or .ply file given!\n");
    printf ("Usage: %s <file.pcd> <scale>\n",argv[0]);
    exit (0);
  }
  
  //Program behavior
  if (pcl::console::find_switch (argc, argv, "--vis")){
    show_vis_ = true;
  }
  
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1){
    if (used_algorithm.compare ("Greedy") == 0){
	use_greedy_ = true;
    }else if (used_algorithm.compare ("Grid") == 0){
	use_grid_ = true;   
    }else if (used_algorithm.compare ("MarchingCubes") == 0){
	use_marchingcube_ = true;   
    }else if (used_algorithm.compare ("Poisson") == 0){
	use_poisson_ = true;   
    }
  }else{
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
  }
  
  //General parameters
  pcl::console::parse_argument (argc, argv, "--leaf_size", leaf_size_);
  //Poisson argumants
  pcl::console::parse_argument (argc, argv, "--poisson_depth", poisson_depth_);
  pcl::console::parse_argument (argc, argv, "--poisson_solver_divide", poisson_solver_divide_);
  pcl::console::parse_argument (argc, argv, "--poisson_iso_divide", poisson_iso_divide_);
  pcl::console::parse_argument (argc, argv, "--poisson_point_weight", poisson_point_weight_);
  pcl::console::parse_argument (argc, argv, "--marching_leaf_size", marching_leafSize_);
  pcl::console::parse_argument (argc, argv, "--marching_isolevel", marching_isoLevel_);

}

// Scale a point cloud
int main (int argc, char **argv)
{
  
  parseCommandLine(argc,argv);
 
  std::vector<int> pcdfiles = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  std::vector<int> plyfiles = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
  PointCloudNT::Ptr cloud_in(new PointCloudNT()); 
  
  
  if(!pcdfiles.empty()){
    print_highlight (stderr, "Loading .pcd file "); print_value (stderr, "%s\n", argv[1]);
    if(pcl::io::loadPCDFile(argv[pcdfiles[0]],*cloud_in) < 0){
      print_error(stderr,"Could not load:"); print_value (stderr, "%s\n", argv[1]);
      return 0;
    }
  }
  if(!plyfiles.empty()){
    if(!loadCloud_ply(argv[plyfiles[0]],*cloud_in)){
      print_error(stderr,"Could not load:"); print_value (stderr, "%s\n", argv[1]);
      return 0;
    }
  }
  
  pcl::PolygonMesh mesh;
  dti::surface::ReconstructPointCloud rec;
   if(use_greedy_){
    print_highlight ("Reconstruct using Greedy Projection Triangulation!\n"); 
    mesh = rec.GreedyProjectionTriangulation(cloud_in,greedy_search);
  }else if(use_grid_){
    print_highlight ("Reconstruct using Grid Projection!\n"); 
    mesh = rec.GridProjection(cloud_in);
  }else if(use_marchingcube_){
    print_highlight ("Reconstruct using Marching Cubes!\n"); 
    mesh = rec.MarchingCubes(cloud_in, marching_leafSize_,marching_isoLevel_);
  }else if(use_poisson_){
    print_highlight ("Reconstruct using Poisson!\n"); 
    rec.poisson(cloud_in,mesh,poisson_depth_,poisson_solver_divide_,poisson_iso_divide_,poisson_point_weight_);
  }
  
  pcl::console::print_highlight("Reconstructed mesh has "); pcl::console::print_value("%d", mesh.polygons.size()); pcl::console::print_info(" polygones and "); pcl::console::print_value("%d", mesh.cloud.data.size()); pcl::console::print_info(" points!\n");
  
  if(plyfiles.size() > 1 || pcdfiles.size() > 1){
   pcl::console::print_info("Saving reconstructed mesh as .ply to: ");pcl::console::print_value("%s", argv[plyfiles[1]]); pcl::console::print_info(" \n");
   TicToc tt;
   tt.tic ();

   print_highlight ("Saving "); print_value ("%s ", argv[plyfiles[1]]);
   pcl::io::savePLYFile (argv[plyfiles[1]], mesh);
   print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", mesh.cloud.width * mesh.cloud.height); print_info (" points & ");
   print_value ("%d", mesh.polygons.size()); print_info (" polygones]\n");
    
  }
  
  if(show_vis_){
   
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis (new pcl::visualization::PCLVisualizer ("Reconstruction"));
    vis->addPolygonMesh(mesh);
 
    vis->spin();
    
  }

 
  

  return (0);
}
