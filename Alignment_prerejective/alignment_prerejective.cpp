#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp_nl.h>


// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
//typedef pcl::SHOT352 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
//typedef pcl::SHOTEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

std::string model_filename_;
std::string scene_filename_;

float leaf(0.006f);
int num_samples(3);
int corr_random(2);
float sim_thres(0.75f);
float inlier_fraction(0.20f);
bool remove_plane(false);
void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*          Alignment pre-rejective pose estimation - Usage Guide          *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <model.pcd> <scene.pcd> [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     This menu" << std::endl;
  std::cout << "     -l:                     Leaf size (default: 0.005)" << std::endl;
  std::cout << "     -s:                     Number of points to sample for generating/prerejecting a pose (default: 3)." << std::endl;
  std::cout << "     -cr:                    Correspondence Randomness - Number of nearest features to use (default: 2)." << std::endl;
  std::cout << "     -st:                    Similarity Threshold - Polygonal edge length similarity threshold (default: 0.9)" << std::endl;
 // std::cout << "     -d:		      Max Correspondence Distance - Inlier threshold" << std::endl;
  std::cout << "     -i:	      	      Inlier Fraction - Required inlier fraction for accepting a pose hypothesis(default 0.25)" << std::endl;
  std::cout << "     --plane:         	      Remove dominante plane (default: false)" << std::endl;
 // std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
 // std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
 // std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
 // std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "--plane"))
  {
    remove_plane = true;
  }
 /* if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }
*/
  //General parameters
  pcl::console::parse_argument (argc, argv, "-l", leaf);
  pcl::console::parse_argument (argc, argv, "-s", num_samples);
  pcl::console::parse_argument (argc, argv, "-cr", corr_random);
  pcl::console::parse_argument (argc, argv, "-st", sim_thres);
  pcl::console::parse_argument (argc, argv, "-i", inlier_fraction);
 // pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
  
}

bool removePlane(PointCloudT::Ptr &src_cloud, PointCloudT::Ptr &target_cloud, double dist_threads){

  Eigen::Vector3f axis = Eigen::Vector3f(0.0,1.0,0.0);
    
   pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

   // Create the segmentation object
   pcl::SACSegmentation<PointNT> seg;
   // Optional
   seg.setOptimizeCoefficients (true);
   // Mandatory
   seg.setModelType (pcl::SACMODEL_PLANE);
   seg.setMethodType (pcl::SAC_RANSAC);
   seg.setDistanceThreshold (dist_threads);
   //seg.setAxis(axis);
   //seg.setEpsAngle(  30.0f * (M_PI/180.0f) );

   seg.setInputCloud (src_cloud);
   seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
   {
     PCL_ERROR ("Could not estimate a planar model for the given dataset.");
     return false;
   }
   
   //*********************************************************************//
   //	Extract Indices
   /**********************************************************************/

   PointCloudT::Ptr cloud_f (new PointCloudT);
  // Create the filtering object
  pcl::ExtractIndices<PointNT> extract;
  // Extract the inliers
  extract.setInputCloud (src_cloud);
  extract.setIndices(inliers);
  // Create the filtering object
  extract.setNegative (true);
  extract.setKeepOrganized(false);
  extract.filter (*cloud_f);
pcl::io::savePCDFile("/home/thso/pe_plane.pcd",*cloud_f);
  pcl::copyPointCloud(*cloud_f, *target_cloud);

  return true;
}

// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
   parseCommandLine (argc, argv);
  
  // Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr object_aligned_ICP (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
   PointCloudT::Ptr full_scene (new PointCloudT);
  FeatureCloudT::Ptr object_features (new FeatureCloudT);
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);
  
  // Get input object and scene
  if (argc < 3)
  {
     showHelp (argv[0]);
    //pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    return (1);
  }
  
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointNT> (model_filename_, *object) < 0 ||
      pcl::io::loadPCDFile<PointNT> (scene_filename_, *scene) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }

  pcl::console::print_highlight ("Loaded object cloud with %d points...\n", object->points.size());
  pcl::console::print_highlight ("Loaded scene cloud with %d points...\n", scene->points.size());
  
   pcl::console::print_highlight ("Loaded scene cloud with %d points...\n", scene->points.size());
  // Downsample
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;

  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);
  grid.filter (*object);
  pcl::console::print_highlight ("Object clouds contain %d points...\n", object->points.size());
  grid.setInputCloud (scene);
  grid.filter (*scene);
  pcl::console::print_highlight ("Scene clouds contain %d points...\n", scene->points.size());
  
  // Estimate normals for scene
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;
  nest.setRadiusSearch (0.01);
 // nest.setKSearch (10);  
 // nest.setInputCloud (object);
 // nest.compute (*object);
  nest.setInputCloud (scene);
  nest.compute (*scene);
  
   if(remove_plane){
      pcl::console::print_highlight ("Removing dominante plane...\n");
      if(!removePlane(scene,scene, 0.005))
	  return (1);
  }
  
  // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (0.025);
//  fest.setKSearch(20);
  fest.setInputCloud (object);
  fest.setInputNormals (object);
  fest.compute (*object_features);
  fest.setInputCloud (scene);
  fest.setInputNormals (scene);
  fest.compute (*scene_features);
  
   pcl::console::print_highlight ("%d object features computed... \n", object_features->points.size());
   pcl::console::print_highlight ("%d scene features computed... \n", object_features->points.size());
  
  // Perform alignment
  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
  align.setInputSource (object);
  align.setSourceFeatures (object_features);
  align.setInputTarget (scene);
  align.setTargetFeatures (scene_features);
 // align.setMaximumIterations (10000); // Number of RANSAC iterations
  align.setNumberOfSamples (num_samples); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (corr_random); // Number of nearest features to use
  align.setSimilarityThreshold (sim_thres); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (1.5f * leaf); // Inlier threshold
  align.setInlierFraction (inlier_fraction); // Required inlier fraction for accepting a pose hypothesis
  // align.setRANSACOutlierRejectionThreshold(0.01); //Default = 0.05
 // align.setEuclideanFitnessEpsilon(0.005);
  {
    pcl::ScopeTime t("Alignment");
    align.align (*object_aligned);
  }
  
  
  


  
  if (align.hasConverged ())
  {
    bool ICP = false;
    if(ICP){
	  // Align
	  pcl::console::print_highlight ("Running ICP...\n");
	  pcl::IterativeClosestPoint<PointNT, PointNT> reg;
	  reg.setInputSource(object_aligned);
	  reg.setInputTarget (scene);
	//  reg.setTransformationEpsilon (1e-9);
	//  reg.setEuclideanFitnessEpsilon(1e-9);
	  reg.setMaxCorrespondenceDistance (5 * leaf);
	  reg.align (*object_aligned_ICP);
	  
	 PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", leaf);
        reg.setMaximumIterations(100);
        reg.setMaxCorrespondenceDistance(leaf);
        reg.align(*object_aligned_ICP, reg.getFinalTransformation());
	 
	  if(reg.hasConverged()){
	
	      Eigen::Matrix4f transformation_icp = reg.getFinalTransformation ();
	      pcl::console::print_info ("ICP Transformation:\n");
	      pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation_icp (0,0), transformation_icp (0,1), transformation_icp (0,2));
	      pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation_icp (1,0), transformation_icp (1,1), transformation_icp (1,2));
	      pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation_icp (2,0), transformation_icp (2,1), transformation_icp (2,2));
	      pcl::console::print_info ("\n");
	      pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation_icp (0,3), transformation_icp (1,3), transformation_icp (2,3));
	      pcl::console::print_info ("ICP Fitnessscore: %f", reg.getFitnessScore());

	  }else{
	       pcl::console::print_error ("ICP has not converged!\n");
	  }
    }
  
    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
    
    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
 
   if(ICP){
     visu.addPointCloud (object_aligned_ICP, ColorHandlerT (object_aligned_ICP, 255.0, 0.0, 0.0), "object_aligned_ICP");
   }else{
        visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
   }
    visu.spin ();
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
    return (1);
  }
  
  return (0);
}
