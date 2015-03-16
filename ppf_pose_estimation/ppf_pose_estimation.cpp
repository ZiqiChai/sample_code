#include <pcl/features/ppf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/normal_refinement.h>

#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/icp_nl.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/console/parse.h>

using namespace pcl;
using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;


float dist_step_ (0.01f); //distance between points in the point pair feature
float dist_angle_(12.0f);
float pos_diff_ (0.1f); //position difference for the clustering algorithm
float rot_diff_ (30.0f); //orientation difference for the clustering algorithm

float leaf_size_ (0.008f);
float plane_thres_ (0.005f);

bool use_icp_ (false);
bool show_scene_normals_ (false);
bool show_model_normals_ (false);
bool normals_changed_(false);
float normals_scale_ = 0.01;

//Scene sampling distance relative to the diameter of the surface model.
unsigned int ref_sampling_rate(25); //25% of the scene point are used as reference points
float normal_estimation_search_radius = 0.01f;

std::string model_filename_;
std::string scene_filename_;

/** \brief Callback for setting options in the visualizer via keyboard.
 *  \param[in] event_arg Registered keyboard event  */
void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event_arg,
                       void*)
{
  int key = event_arg.getKeyCode ();

  if (event_arg.keyUp ())
    switch (key)
    {
      case (int) '1': //show scene_normals
        show_scene_normals_ = !show_scene_normals_;
        normals_changed_ = true;
        break;
	
     case (int) '2': //show model_normals
        show_model_normals_ = !show_model_normals_;
        normals_changed_ = true;
        break;
      case (int) '3': 
        normals_scale_ *= 1.25;
        normals_changed_ = true;
        break;
      case (int) '4':
        normals_scale_ *= 0.8;
        normals_changed_ = true;
        break;
   
      default:
        break;
    }
}

PointCloud<PointNormal>::Ptr
subsample(PointCloud<PointNT>::Ptr cloud)
{
  
  const Eigen::Vector4f subsampling_leaf_size (leaf_size_, leaf_size_, leaf_size_, 0.0f);
  PointCloud<PointNT>::Ptr cloud_subsampled (new PointCloud<PointNT> ());
  VoxelGrid<PointNT> subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (subsampling_leaf_size);
  subsampling_filter.filter (*cloud_subsampled);

 
 
  PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
  return cloud_subsampled;
}

PointCloud<PointNormal>::Ptr
subsampleAndCalculateNormals (PointCloud<PointNT>::Ptr cloud)
{
  const Eigen::Vector4f subsampling_leaf_size (leaf_size_, leaf_size_, leaf_size_, 0.0f);
  PointCloud<PointNT>::Ptr cloud_subsampled (new PointCloud<PointNT> ());
  VoxelGrid<PointNT> subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (subsampling_leaf_size);
  subsampling_filter.filter (*cloud_subsampled);

  PointCloud<Normal>::Ptr cloud_subsampled_normals (new PointCloud<Normal> ());
 
  /*NormalEstimationOMP<PointT, Normal> normal_estimation_filter;
  normal_estimation_filter.setInputCloud (cloud_subsampled);
  search::KdTree<PointT>::Ptr search_tree (new search::KdTree<PointT>);
  normal_estimation_filter.setSearchMethod (search_tree);
  normal_estimation_search_radius = 1.5 * leaf_size_;
  normal_estimation_filter.setRadiusSearch (normal_estimation_search_radius);
  //normal_estimation_filter.setKSearch(4);
  normal_estimation_filter.setViewPoint(0,0,1);
  normal_estimation_filter.setNumberOfThreads(2);
  normal_estimation_filter.compute (*cloud_subsampled_normals);
*/
  
  pcl::PointCloud<Normal>::Ptr normals_refined(new pcl::PointCloud<Normal>);

  // Search parameters
  const int k = 5;
  std::vector<std::vector<int> > k_indices;
  std::vector<std::vector<float> > k_sqr_distances;
  // Run search
  pcl::search::KdTree<PointNT> search;
  search.setInputCloud (cloud_subsampled);
  search.nearestKSearch (*cloud_subsampled, std::vector<int> (), k, k_indices, k_sqr_distances);
  // Use search results for normal estimation
  pcl::NormalEstimationOMP<PointNT, Normal> ne;
  for (unsigned int i = 0; i < cloud_subsampled->size (); ++i){
  Normal normal;
  ne.computePointNormal (*cloud_subsampled, k_indices[i],
                         normal.normal_x, normal.normal_y, normal.normal_z, normal.curvature);
  pcl::flipNormalTowardsViewpoint (cloud_subsampled->at(i), 0, 0, 1,
                                   normal.normal_x, normal.normal_y, normal.normal_z);
  cloud_subsampled_normals->push_back (normal);
  }

  // Run refinement using search results
  pcl::NormalRefinement<Normal> nr (k_indices, k_sqr_distances);
  nr.setInputCloud (cloud_subsampled_normals);
  nr.filter (*normals_refined);
  
  PointCloud<PointNT>::Ptr cloud_subsampled_with_normals (new PointCloud<PointNT> ());
  //concatenateFields (*cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);
  concatenateFields (*cloud_subsampled, *normals_refined, *cloud_subsampled_with_normals);

  PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
  return cloud_subsampled_with_normals;
}

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             ppf_registration - Usage Guide              		     *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <model_filename.pcd> <scene_filename.pcd> [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:			Show this help." << std::endl;
  std::cout << "     -i:			Use ICP fine registration." << std::endl;
 // std::cout << "     -c:                     Show used correspondences." << std::endl;
 // std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
 // std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --leaf_size:		Voxel_grid leaf_size (default 0.02)" << std::endl;
  std::cout << "     --dist_step:         	Distance_discretization_step (default 0.05)" << std::endl;
  std::cout << "     --dist_angle:         	Orientaion_discretization_step (default 12.0)" << std::endl;
  std::cout << "     --plane_thres:		Plane removal distance threshold  (default 0.04)" << std::endl;
  std::cout << "     --pos_diff:		Position difference for the clustering algorithm(default 0.1)" << std::endl;
  std::cout << "     --rot_diff:		Rotation difference for the clustering algorithm(default 10 degrees)" << std::endl;
  std::cout << "     --ref_sampling_rate:	Reference sampling rate(default: 25)" << std::endl;
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
  if (pcl::console::find_switch (argc, argv, "-i"))
  {
    use_icp_ = true;
  }
  

  //General parameters
  pcl::console::parse_argument (argc, argv, "--leaf_size", leaf_size_);
  pcl::console::parse_argument (argc, argv, "--dist_step", dist_step_);
  pcl::console::parse_argument (argc, argv, "--dist_angle", dist_angle_);
  pcl::console::parse_argument (argc, argv, "--plane_thres", plane_thres_);
  pcl::console::parse_argument (argc, argv, "--pos_diff", pos_diff_);
  pcl::console::parse_argument (argc, argv, "--rot_diff", rot_diff_);
  pcl::console::parse_argument (argc, argv, "--ref_sampling_rate", ref_sampling_rate);
  
}

int
main (int argc, char** argv)
{
    parseCommandLine (argc, argv);

  /// read
  std::vector<int> dummy;
  /// read point clouds from HDD
  PCL_INFO ("Loading scene ...\n");
  pcl::PCLPointCloud2 pcl2_scene_cloud;
  PointCloud<PointNT>::Ptr cloud_scene (new PointCloud<PointNT> ());
  PCDReader reader;
  if(reader.read (scene_filename_, pcl2_scene_cloud) != 0){
   PCL_ERROR("Could not load scene!!\n");  
  }
  pcl::fromPCLPointCloud2 (pcl2_scene_cloud, *cloud_scene); 
  pcl::removeNaNFromPointCloud(*cloud_scene, *cloud_scene, dummy);
  PCL_INFO ("Scene loaded: %s\n", scene_filename_.c_str());

  PCL_INFO ("Loading model ...\n");
  vector<PointCloud<PointNT>::Ptr > cloud_models;
  pcl::PCLPointCloud2 pcl2_model_cloud;
  PointCloud<PointNT>::Ptr cloud (new PointCloud<PointNT> ());
  
   if(reader.read (model_filename_, pcl2_model_cloud) != 0){
   PCL_ERROR("Could not load model!!\n");  
  }
  
  pcl::fromPCLPointCloud2 (pcl2_model_cloud, *cloud); 
  pcl::removeNaNFromPointCloud(*cloud, *cloud, dummy);
  cloud_models.push_back (cloud);
  PCL_INFO ("Model read: %s\n", argv[1]);


   Eigen::Vector3f axis = Eigen::Vector3f(0.0,1.0,0.0);
  pcl::SACSegmentation<PointNT> seg;
  pcl::ExtractIndices<PointNT> extract;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (plane_thres_);
  seg.setAxis(axis);
  extract.setNegative (true);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  unsigned nr_points = unsigned (cloud_scene->points.size ());
  while (cloud_scene->points.size () > 0.3 * nr_points)
  {
    seg.setInputCloud (cloud_scene);
    seg.segment (*inliers, *coefficients);
    PCL_INFO ("Plane inliers: %u\n", inliers->indices.size ());
    if (inliers->indices.size () < 50000) break;

    extract.setInputCloud (cloud_scene);
    extract.setIndices (inliers);
    extract.filter (*cloud_scene);
  }

  PointCloud<PointNT>::Ptr cloud_scene_input = subsampleAndCalculateNormals (cloud_scene);
  vector<PointCloud<PointNT>::Ptr > cloud_models_with_normals;
   

  PCL_INFO ("Training model(s) ...\n");
  vector<PPFHashMapSearch::Ptr> hashmap_search_vector;
  for (size_t model_i = 0; model_i < cloud_models.size (); ++model_i)
  {
    PointCloud<PointNT>::Ptr cloud_model_input;
    if(pcl::getFieldIndex(pcl2_model_cloud,"normal_x") >= 0){
      PCL_INFO("Model has normals... No normal estimation needed!!\n");
       cloud_model_input = subsample(cloud_models[model_i]);
    }else{
       PCL_INFO("Model has no normals... Estimating normals!!\n");
     cloud_model_input = subsampleAndCalculateNormals (cloud_models[model_i]);
    }
      
    
    cloud_models_with_normals.push_back (cloud_model_input);

    PointCloud<PPFSignature>::Ptr cloud_model_ppf (new PointCloud<PPFSignature> ());
    PPFEstimation<PointNT, PointNT, PPFSignature> ppf_estimator;
    ppf_estimator.setInputCloud (cloud_model_input);
    ppf_estimator.setInputNormals (cloud_model_input);
    ppf_estimator.compute (*cloud_model_ppf);
  
    PPFHashMapSearch::Ptr hashmap_search (new PPFHashMapSearch (dist_angle_ / 180.0f * float (M_PI),
                                                                 dist_step_));
    hashmap_search->setInputFeatureCloud (cloud_model_ppf);
    hashmap_search_vector.push_back (hashmap_search);
  }

  visualization::PCLVisualizer viewer ("PPF Pose Estimation - Results");
  viewer.setBackgroundColor (0, 0, 0);
  viewer.registerKeyboardCallback (keyboardEventOccurred, 0);
  viewer.addPointCloud (cloud_scene,ColorHandlerT (cloud_scene, 255.0, 0.0, 0.0), "scene");
  viewer.spinOnce (10);
  
  PCL_INFO ("Registering models to scene ...\n");

   PointCloud<PointNormal>::Ptr cloud_model_with_normals_trfm (new PointCloud<PointNormal>);
  for (size_t model_i = 0; model_i < cloud_models.size (); ++model_i)
  {

    PPFRegistration<PointNT, PointNT> ppf_registration;
    // set parameters for the PPF registration procedure
    ppf_registration.setSceneReferencePointSamplingRate (ref_sampling_rate);
    ppf_registration.setPositionClusteringThreshold (pos_diff_); //0.2 
    ppf_registration.setRotationClusteringThreshold (rot_diff_ / 180.0f * float (M_PI));
    ppf_registration.setSearchMethod (hashmap_search_vector[model_i]);
    ppf_registration.setInputSource (cloud_models_with_normals[model_i]);
    ppf_registration.setInputTarget (cloud_scene_input);

    PointCloud<PointNT> cloud_output_subsampled;
    ppf_registration.align (cloud_output_subsampled);

   /* PointCloud<PointT>::Ptr cloud_output_subsampled_xyz (new PointCloud<PointT> ());
    for (size_t i = 0; i < cloud_output_subsampled.points.size (); ++i)
      cloud_output_subsampled_xyz->points.push_back ( PointT (cloud_output_subsampled.points[i].x, cloud_output_subsampled.points[i].y, cloud_output_subsampled.points[i].z));
*/

    Eigen::Matrix4f mat = ppf_registration.getFinalTransformation ();
    Eigen::Affine3f final_transformation (mat);

    
    //  io::savePCDFileASCII ("output_subsampled_registered.pcd", cloud_output_subsampled);

    PointCloud<PointNT>::Ptr cloud_output (new PointCloud<PointNT> ());
   double  icp_fitness_score = 0;
    if(use_icp_){
     PCL_INFO ("Running ICP fine registration ...\n");
     PointCloud<PointNormal>::Ptr cloud_output_icp (new PointCloud<PointNormal> ());
     
     pcl::transformPointCloudWithNormals<PointNT> (*cloud_models_with_normals[model_i], *cloud_model_with_normals_trfm, final_transformation);
     pcl::transformPointCloud (*cloud_models[model_i], *cloud_models[model_i], final_transformation);
     
     pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
     reg.setTransformationEpsilon (1e-9);
     reg.setEuclideanFitnessEpsilon(1e-9);
     reg.setMaxCorrespondenceDistance (3.0f * leaf_size_);
     // reg.setRANSACOutlierRejectionThreshold(0.01);
       //reg.setRANSACIterations(10000);
       reg.setInputSource(cloud_model_with_normals_trfm);
       reg.setInputTarget (cloud_scene_input);
       reg.align (*cloud_output_icp);
       pcl::copyPointCloud(*cloud_output_icp, *cloud_output);
       icp_fitness_score = reg.getFitnessScore();
       mat = reg.getFinalTransformation();
       final_transformation = mat;
    }

    pcl::transformPointCloud (*cloud_models[model_i], *cloud_output, final_transformation);
    pcl::transformPointCloudWithNormals<PointNT>(*cloud_models_with_normals[model_i], *cloud_model_with_normals_trfm, final_transformation);
  
    pcl::console::print_info ("Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", mat (0,0), mat (0,1), mat (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", mat (1,0), mat (1,1), mat (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", mat (2,0), mat (2,1), mat (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", mat (0,3), mat (1,3), mat (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("PPF Fitness Score: %f\n", ppf_registration.getFitnessScore());
    if(use_icp_) pcl::console::print_info ("ICP Fitness Score: %f\n", icp_fitness_score);
    

    stringstream ss; ss << "model_" << model_i;
    viewer.addPointCloud (cloud_output, ColorHandlerT (cloud_output, 0.0, 255.0, 0.0), ss.str ());
    PCL_INFO ("Showing model %s\n", ss.str ().c_str ());
   
  }

  PCL_INFO ("All models have been registered!\n");


  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    
   if (normals_changed_){
          viewer.removePointCloud ("scene_normals");
	  viewer.removePointCloud ("model_normals");
        if (show_scene_normals_){
	
          viewer.addPointCloudNormals<PointNT>(cloud_scene_input, 1, normals_scale_, "scene_normals");
	  normals_changed_ = false;
        }
        
        if (show_model_normals_){
	 viewer.addPointCloudNormals<PointNT>(cloud_model_with_normals_trfm, 1, normals_scale_, "model_normals");
	  normals_changed_ = false;
        }
      }
  }

  return 0;
}
