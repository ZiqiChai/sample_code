#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/common/distances.h>
#include <pcl/correspondence.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/ppf.h>
#include <pcl/filters/normal_refinement.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include <pcl/registration/transformation_estimation_svd.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <covis/feature/normal_correction_manifold.h>

using namespace pcl;
using namespace std;

#define PI 3.14159265359

typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

typedef pcl::FPFHSignature33 FeatureT;
//typedef pcl::SHOT352 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
//typedef pcl::SHOTEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

/// Prerejective RANSAC parameters
int num_samples_(3);
int corr_random_(2);
float sim_thres_(0.75f);
float inlier_fraction_(0.20f);

///Corresponding grouping parameters
//float cg_model_ss_ (0.01f);
//float cg_scene_ss_ (0.03f);
float cg_rf_rad_ (0.015f);
float cg_descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);
bool use_hough_ (true);
/*int icp_max_iter_ (5);
float icp_corr_distance_ (0.005f);
float hv_clutter_reg_ (5.0f);
float hv_inlier_th_ (0.005f);
float hv_occlusion_th_ (0.01f);
float hv_rad_clutter_ (0.03f);
float hv_regularizer_ (3.0f);
float hv_rad_normals_ (0.05);
bool hv_detect_clutter_ (true);
*/
///PPF_registration parameters
float dist_step_ (0.01f); //distance between points in the point pair feature
float dist_angle_(12.0f);
float pos_diff_ (0.1f); //position difference for the clustering algorithm
float rot_diff_ (30.0f); //orientation difference for the clustering algorithm
//Scene sampling distance relative to the diameter of the surface model.
unsigned int ref_sampling_rate(25); //25% of the scene point are used as reference points

/// Control parameters 
bool use_RANSAC_ (false);
bool use_prereject_ (true);
bool use_PPF_ (false);
bool use_CG_ (false);
bool use_icp_ (false);
bool show_keypoints_(false);
bool show_correspondences_(false);
bool show_vis_ (false);
bool show_scene_normals_ (false);
bool show_model_normals_ (false);
bool show_ground_truth_ (false);
bool show_ground_truth_changed_ (false);
bool show_frames_ (false);
bool show_frames_changed_ (false);
bool remove_plane(false);
bool save_aligned_cloud(false);
bool normals_changed_(false);
float normals_scale_ = 0.01;


///Common parameters
float leaf_size_ (0.006f);
float plane_thres_ (0.005f);
std::string gt_path("");
bool gt_ignore_z_rot_(false);
bool gt_ignore_y_rot_(false);
bool gt_ignore_x_rot_(false);
std::string flip_around_x_(""); 

std::vector<std::string> model_filenames_;
std::string scene_filename_;

/** \brief Callback for setting options in the visualizer via keyboard.
 *  \param[in] event_arg Registered keyboard event  */
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event_arg,void*){
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
	
      case (int) '5':
        show_ground_truth_ = !show_ground_truth_;
	show_ground_truth_changed_ = true;
        break;
   
      case (int) '6':
        show_frames_ = !show_frames_;
	show_frames_changed_ = true;
        break;
      default:
        break;
    }
}

void subsample(PointCloud<PointNT>::Ptr cloud){//, PointCloud<PointNT>::Ptr dst){
  
  PointCloud<PointNT>::Ptr cloud_subsampled (new PointCloud<PointNT> ());
  VoxelGrid<PointNT> subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
  subsampling_filter.filter (*cloud);
 // PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
 // copyPointCloud(*cloud_subsampled, *dst);
}

void subsampleAndCalculateNormals (PointCloud<PointNT>::Ptr cloud,  bool sample, bool use_normal_correction)//, PointCloud<PointNT>::Ptr dst)
{
 // PointCloud<PointNT>::Ptr cloud_subsampled (new PointCloud<PointNT> ());
 if(sample){
  VoxelGrid<PointNT> subsampling_filter;
  subsampling_filter.setInputCloud (cloud);
  subsampling_filter.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
  subsampling_filter.filter (*cloud);
  }
//  PointCloud<PointNT>::Ptr cloud_subsampled_normals (new PointCloud<PointNT> ());
 
 NormalEstimationOMP<PointNT, PointNT> normal_estimation_filter;
  normal_estimation_filter.setInputCloud (cloud);
  search::KdTree<PointNT>::Ptr search_tree (new search::KdTree<PointNT>);
  normal_estimation_filter.setSearchMethod (search_tree);
 // normal_estimation_filter.setRadiusSearch (1.5f * leaf_size_);
  normal_estimation_filter.setRadiusSearch (2.0f * leaf_size_);
  //normal_estimation_filter.setKSearch(4);
  normal_estimation_filter.setViewPoint(0,0,1);
//  normal_estimation_filter.setNumberOfThreads(2);
  normal_estimation_filter.compute (*cloud);
  
  if(use_normal_correction){
//  pcl::io::savePCDFile("model_wo_corrected_normals.pcd", *cloud);
    covis::feature::NormalCorrectionManifold<PointNT> ncm;
   // ncm.setK(450);
   ncm.setUseKNN(false);
    //ncm.setRadius(1.5f * leaf_size_);
    ncm.setRadius(2.0f * leaf_size_);
    ncm.compute(*cloud);
// pcl::io::savePCDFile("model_w_corrected_normals.pcd", *cloud);
  }
/*  
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
  
//  PointCloud<PointNT>::Ptr cloud_subsampled_with_normals (new PointCloud<PointNT> ());
  //concatenateFields (*cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);
  
  */
 // concatenateFields (*cloud_subsampled, *cloud_subsampled_normals, *dst);

//  PCL_INFO ("Cloud dimensions before / after subsampling: %u / %u\n", cloud->points.size (), cloud_subsampled->points.size ());
//  return cloud_subsampled_with_normals;
}

void showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             pose_estimation_app - Usage Guide              		     *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <scene_filename.pcd> N*<model_filename.pcd> [Options]" << std::endl << std::endl;
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
  
  std::cout << "********************** PPF Parameters *************************************" << std::endl << std::endl;
  std::cout << "     --dist_step:         			Distance_discretization_step (default 0.01)" << std::endl;
  std::cout << "     --dist_angle:         			Orientaion_discretization_step (default 12.0)" << std::endl;
  std::cout << "     --pos_diff:				Position difference for the clustering algorithm(default 0.1)" << std::endl;
  std::cout << "     --rot_diff:				Rotation difference for the clustering algorithm(default 30 degrees)" << std::endl;
  std::cout << "     --ref_sampling_rate:			Reference sampling rate(default: 25)" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;

  std::cout << "********************** Prereject Parameters *************************************" << std::endl << std::endl;
  std::cout << "     --prereject_samples:			Number of points to sample for generating/prerejecting a pose (default: 3)." << std::endl;
  std::cout << "     --corr_random:				Correspondence Randomness - Number of nearest features to use (default: 2)." << std::endl;
  std::cout << "     --similarity_thres:			Similarity Threshold - Polygonal edge length similarity threshold (default: 0.9)" << std::endl;
 // std::cout << "     -d:		      Max Correspondence Distance - Inlier threshold" << std::endl;
  std::cout << "     --inlier:					Inlier Fraction - Required inlier fraction for accepting a pose hypothesis(default 0.25)" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;

  std::cout << "********************* Corresponding Grouping (GC) Parameters ********************" << std::endl << std::endl;
  std::cout << "     --GC_algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
 // std::cout << "     --model_ss:         Model uniform sampling radius (default 0.01)" << std::endl;
 // std::cout << "     --scene_ss:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad:           			Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad:        			Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size:          			Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh:        			Clustering threshold (default 5)" << std::endl << std::endl;
 /* std::cout << "     --icp_max_iter val:          		ICP max iterations number (default " << icp_max_iter_ << ")" << std::endl;
  std::cout << "     --icp_corr_distance val:     		ICP correspondence distance (default " << icp_corr_distance_ << ")" << std::endl << std::endl;
  std::cout << "     --hv_clutter_reg val:        		Clutter Regularizer (default " << hv_clutter_reg_ << ")" << std::endl;
  std::cout << "     --hv_inlier_th val:          		Inlier threshold (default " << hv_inlier_th_ << ")" << std::endl;
  std::cout << "     --hv_occlusion_th val:       		Occlusion threshold (default " << hv_occlusion_th_ << ")" << std::endl;
  std::cout << "     --hv_rad_clutter val:        		Clutter radius (default " << hv_rad_clutter_ << ")" << std::endl;
  std::cout << "     --hv_regularizer val:        		Regularizer value (default " << hv_regularizer_ << ")" << std::endl;
  std::cout << "     --hv_rad_normals val:        		Normals radius (default " << hv_rad_normals_ << ")" << std::endl;
  std::cout << "     --hv_detect_clutter val:     		TRUE if clutter detect enabled (default " << hv_detect_clutter_ << ")" << std::endl << std::endl;
  
  */
  
  std::cout << "***************************************************************************" << std::endl << std::endl;

  
  //  std::cout << "     -s:                     Number of points to sample for generating/prerejecting a pose (default: 3)." << std::endl;
 // std::cout << "     -cr:                    Correspondence Randomness - Number of nearest features to use (default: 2)." << std::endl;
 // std::cout << "     -st:                    Similarity Threshold - Polygonal edge length similarity threshold (default: 0.9)" << std::endl;
 // std::cout << "     -d:		      Max Correspondence Distance - Inlier threshold" << std::endl;
 // std::cout << "     -i:	      	      Inlier Fraction - Required inlier fraction for accepting a pose hypothesis(default 0.25)" << std::endl;

  
}

void parseCommandLine (int argc, char *argv[])
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
  if (filenames.size () < 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  scene_filename_ = argv[filenames[0]];
  for(unsigned int i = 1; i < filenames.size(); i++){
   model_filenames_.push_back(argv[filenames[i]]);  
  }

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "--icp"))
  {
    use_icp_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "--vis"))
  {
    show_vis_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "--plane"))
  {
    remove_plane = true;
  }
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "--save"))
  {
    save_aligned_cloud = true;
  }
  
   std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("RANSAC") == 0)
    {
      use_prereject_ = false;
      use_RANSAC_ = true;
    }else if (used_algorithm.compare ("CG") == 0)
    {
       std::string used_algorithm;
      if (pcl::console::parse_argument (argc, argv, "--GC_algorithm", used_algorithm) != -1){
	if (used_algorithm.compare ("Hough") == 0){
	    use_hough_ = true;
	}else if (used_algorithm.compare ("GC") == 0){
	    use_hough_ = false;
	}
	else{
	    std::cout << "Wrong algorithm name.\n";
	    showHelp (argv[0]);
	    exit (-1);
	}
  }
      use_prereject_ = false;
      use_CG_ = true;
    }else if (used_algorithm.compare ("Prereject") == 0)
    {
      use_prereject_ = true;
    }else if (used_algorithm.compare ("PPF") == 0)
    {
      use_prereject_ = false;
      use_PPF_ = true;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }
  
  //General parameters
  pcl::console::parse_argument (argc, argv, "--leaf_size", leaf_size_);
  pcl::console::parse_argument (argc, argv, "--gt", gt_path);
  pcl::console::parse_argument (argc, argv, "--gt_ignore_z_rot", gt_ignore_z_rot_);
  pcl::console::parse_argument (argc, argv, "--gt_ignore_y_rot", gt_ignore_y_rot_);
  pcl::console::parse_argument (argc, argv, "--gt_ignore_x_rot", gt_ignore_x_rot_);
  pcl::console::parse_argument (argc, argv, "--gt_flip_around_x", flip_around_x_);

 
  pcl::console::parse_argument (argc, argv, "--dist_step", dist_step_);
  pcl::console::parse_argument (argc, argv, "--dist_angle", dist_angle_);
  pcl::console::parse_argument (argc, argv, "--plane_thres", plane_thres_);
  pcl::console::parse_argument (argc, argv, "--pos_diff", pos_diff_);
  pcl::console::parse_argument (argc, argv, "--rot_diff", rot_diff_);
  pcl::console::parse_argument (argc, argv, "--ref_sampling_rate", ref_sampling_rate);
  
  pcl::console::parse_argument (argc, argv, "--prereject_samples",num_samples_ );
  pcl::console::parse_argument (argc, argv, "--corr_random",corr_random_ );
  pcl::console::parse_argument (argc, argv, "--similarity_thres",sim_thres_ );
  pcl::console::parse_argument (argc, argv, "--inlier",inlier_fraction_ );
  
 // pcl::console::parse_argument (argc, argv, "--model_ss",cg_model_ss_ );
 // pcl::console::parse_argument (argc, argv, "--scene_ss",cg_scene_ss_ );
  pcl::console::parse_argument (argc, argv, "--rf_rad",cg_rf_rad_ );
  pcl::console::parse_argument (argc, argv, "--descr_rad",cg_descr_rad_ );
  pcl::console::parse_argument (argc, argv, "--cg_size",cg_size_ );
  pcl::console::parse_argument (argc, argv, "--cg_thresh",cg_thresh_ );
/*  pcl::console::parse_argument (argc, argv, "--icp_max_iter", icp_max_iter_);
  pcl::console::parse_argument (argc, argv, "--icp_corr_distance", icp_corr_distance_);
  pcl::console::parse_argument (argc, argv, "--hv_clutter_reg", hv_clutter_reg_);
  pcl::console::parse_argument (argc, argv, "--hv_inlier_th", hv_inlier_th_);
  pcl::console::parse_argument (argc, argv, "--hv_occlusion_th", hv_occlusion_th_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_clutter", hv_rad_clutter_);
  pcl::console::parse_argument (argc, argv, "--hv_regularizer", hv_regularizer_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_normals", hv_rad_normals_);
  pcl::console::parse_argument (argc, argv, "--hv_detect_clutter", hv_detect_clutter_);
  */
}

bool icp_pointT(PointCloudT::Ptr &src, PointCloudT::Ptr &tar, PointCloudT::Ptr &aligned, Eigen::Matrix4f &transform){
   
  Eigen::Matrix4f transform_;
  pcl::IterativeClosestPointNonLinear<PointNT,PointNT> icp;
 // PointCloudT tmp;

  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 5 * leaf_size_);
  icp.setInputSource(src);
  icp.setInputTarget(tar);
  icp.setMaximumIterations(300);
  icp.setMaxCorrespondenceDistance(5 * leaf_size_);
  icp.align(*aligned, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();


  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }
  
  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 1 * leaf_size_);
  icp.setMaxCorrespondenceDistance(1 * leaf_size_);
   icp.setMaximumIterations(200);
  icp.align(*aligned, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();


  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }
  
    PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.5 * leaf_size_);
  icp.setMaxCorrespondenceDistance(0.5 * leaf_size_);
    // icp.setMaximumIterations(50);
  icp.align(*aligned, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();


  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }
  
    PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.25 * leaf_size_);
  icp.setMaxCorrespondenceDistance(0.25 * leaf_size_);
  icp.align(*aligned, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();


  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }
  
/*    pcl::console::print_info ("ICP Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (0,0), transform_ (0,1), transform_ (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transform_ (1,0), transform_ (1,1), transform_ (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (2,0), transform_ (2,1), transform_ (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transform_ (0,3), transform_ (1,3), transform_ (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("ICP Fitness Score: %f\n", icp.getFitnessScore()); 
  */  
    transform << transform_;

  return true;
}


float computeError(PointCloudT::Ptr &scene,PointCloudT::Ptr &aligned_cloud){
   float rmse = 0.0f; 
   //float median_error = 0.0f;
   //float mean_error = 0.0f;
   //float std_dev = 0.0f;
   std::vector<float> distances;
   float valid_points = 0.0;
   pcl::console::print_highlight (stderr, "Computing rmse using the nearest neighbor correspondence heuristic.\n");

    KdTreeFLANN<PointNT>::Ptr tree (new KdTreeFLANN<PointNT> ());
    tree->setInputCloud (scene);

    for (size_t point_i = 0; point_i < aligned_cloud->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (aligned_cloud->points[point_i].x) || !pcl_isfinite (aligned_cloud->points[point_i].y) || !pcl_isfinite (aligned_cloud->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (aligned_cloud->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      float dist = squaredEuclideanDistance (aligned_cloud->points[point_i], scene->points[point_nn_i]);
      if(dist < 0.020){ //only consider points closer than 20mm
	rmse += dist;
	distances.push_back(dist);
	valid_points++;
      }
    }
   // rmse = sqrtf(rmse / static_cast<float> (aligned_cloud->points.size ()));
    rmse = sqrtf(rmse / valid_points);
  //  median_error = median(distances);  
  //  mean_error = mean(distances);
  //  std_dev = stdDev(distances);
    
    pcl::console::print_highlight ("RMSE Error: %e (%f mm)\n", rmse, rmse*1000);
  //  pcl::console::print_highlight ("Median Error: %e (%f mm)\n", median_error, median_error*1000);
  //  pcl::console::print_highlight ("Mean Error: %e (%f mm)\n", mean_error, mean_error*1000);
  //  pcl::console::print_highlight ("Std_dev: %e (%f mm)\n", std_dev, std_dev*1000);  
    
    return rmse;
}


bool run_RANSAC( PointCloudT::Ptr &scene,  PointCloudT::Ptr &object, Eigen::Matrix4f &final_transformation, PointCloudT::Ptr &aligned_cloud, bool use_prereject )
{
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr object_aligned_ICP (new PointCloudT);
  FeatureCloudT::Ptr object_features (new FeatureCloudT);
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);
  
   // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (5 * leaf_size_); //4
  //fest.setKSearch(8); //20
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
  align.setMaximumIterations (50000); // Number of RANSAC iterations
  align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (2); // Number of nearest features to use
  if(use_prereject)align.setSimilarityThreshold (sim_thres_); // Polygonal edge length similarity threshold
  else align.setSimilarityThreshold (0.0f); // Polygonal edge length similarity threshold
  //align.setMaxCorrespondenceDistance (1.5f * leaf_size_); // Inlier threshold
  align.setMaxCorrespondenceDistance (2.0f * leaf_size_); // Inlier threshold
  align.setInlierFraction (inlier_fraction_); // Required inlier fraction for accepting a pose hypothesis
  // align.setRANSACOutlierRejectionThreshold(0.01); //Default = 0.05
 // align.setEuclideanFitnessEpsilon(0.005);
  {
    pcl::ScopeTime t("Alignment");
    align.align (*object_aligned);
  }
  
  
  
  if (align.hasConverged ()){
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
    
    final_transformation = transformation;
    copyPointCloud(*object_aligned, *aligned_cloud);
   
    return true;
    
  } else{
    pcl::console::print_error ("Alignment failed!\n");
    return false;
  }
  
}

bool run_ppf_registration(PointCloudT::Ptr &scene,  PointCloudT::Ptr &object, Eigen::Matrix4f &final_transformation, PointCloudT::Ptr &aligned_cloud )
{
 
  vector<PointCloud<PointNT>::Ptr > cloud_models_with_normals;
  
  PCL_INFO ("Training model(s) ...\n");
 // vector<PPFHashMapSearch::Ptr> hashmap_search_vector;

  PointCloud<PPFSignature>::Ptr cloud_model_ppf (new PointCloud<PPFSignature> ());
  PPFEstimation<PointNT, PointNT, PPFSignature> ppf_estimator;
  ppf_estimator.setInputCloud (object);
  ppf_estimator.setInputNormals (object);
  ppf_estimator.compute (*cloud_model_ppf);
  
  PPFHashMapSearch::Ptr hashmap_search (new PPFHashMapSearch (dist_angle_ / 180.0f * float (M_PI),
                                                              dist_step_));
  hashmap_search->setInputFeatureCloud (cloud_model_ppf);
  //hashmap_search_vector.push_back (hashmap_search);
  
  PCL_INFO ("Registering models to scene ...\n");
 
  PPFRegistration<PointNT, PointNT> ppf_registration;
  // set parameters for the PPF registration procedure
  ppf_registration.setSceneReferencePointSamplingRate (ref_sampling_rate);
  ppf_registration.setPositionClusteringThreshold (pos_diff_); //0.2 
  ppf_registration.setRotationClusteringThreshold (rot_diff_ / 180.0f * float (M_PI));
  ppf_registration.setSearchMethod (hashmap_search);
  ppf_registration.setInputSource (object);
  ppf_registration.setInputTarget (scene);

  PointCloud<PointNT> cloud_output_subsampled;
  ppf_registration.align (cloud_output_subsampled);

  Eigen::Matrix4f mat = Eigen::Matrix4f::Identity(); 
  if(ppf_registration.hasConverged()){
    mat = ppf_registration.getFinalTransformation ();
    Eigen::Affine3f final_transformation_ (mat);
 
    pcl::transformPointCloud (*object, *aligned_cloud, final_transformation_);
  }else{
    pcl::console::print_error ("Alignment failed!\n");
   return false;
  }
  
  final_transformation = mat;
 
  pcl::console::print_info ("Transformation:\n");
  pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", mat (0,0), mat (0,1), mat (0,2));
  pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", mat (1,0), mat (1,1), mat (1,2));
  pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", mat (2,0), mat (2,1), mat (2,2));
  pcl::console::print_info ("\n");
  pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", mat (0,3), mat (1,3), mat (2,3));
  pcl::console::print_info ("\n");
  pcl::console::print_info ("PPF Fitness Score: %f\n", ppf_registration.getFitnessScore());
  //computeError(scene,aligned_cloud);
  
  return true;
  
}

bool run_CG(PointCloudT::Ptr &scene_sampled,  PointCloudT::Ptr &object_sampled,
	    PointCloudT::Ptr &full_scene,  PointCloudT::Ptr &full_object,
	    Eigen::Matrix4f &final_transformation, PointCloudT::Ptr &aligned_cloud	){
  
  typedef pcl::SHOT352 DescriptorType;
  typedef pcl::ReferenceFrame RFType;
  
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  
   //
  //  Compute Descriptor for keypoints - we
  //
  pcl::console::print_highlight ("Estimating features...\n");
  pcl::SHOTEstimationOMP<PointNT, PointNT, DescriptorType> descr_est;
  descr_est.setRadiusSearch (cg_descr_rad_);

  descr_est.setInputCloud (object_sampled); //model_keypoints
  descr_est.setInputNormals (object_sampled); //model_normals
//  descr_est.setSearchSurface (full_object);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_sampled); //
  descr_est.setInputNormals (scene_sampled); //
 // descr_est.setSearchSurface (full_scene);
  descr_est.compute (*scene_descriptors);
  
   //
  //  Find Model-Scene Correspondences with KdTree
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
  pcl::console::print_highlight ("Finding correspondences...\n");
  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
  
   //
  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  use_hough_ = true;
  //  Using Hough3D
  if (use_hough_)
  {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointNT, PointNT, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (cg_rf_rad_);

    rf_est.setInputCloud (object_sampled);
    rf_est.setInputNormals (object_sampled);
    rf_est.setSearchSurface (full_object);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_sampled);
    rf_est.setInputNormals (scene_sampled);
    rf_est.setSearchSurface (full_scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);
/*
    clusterer.setInputCloud (object_sampled);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_sampled);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
    */
  }
  else // Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointNT, PointNT> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (object_sampled);
    gc_clusterer.setSceneCloud (scene_sampled);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  //
  //  Output results
  //
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  
   /**
   * Generates clouds for each instances found 
   */
/*  std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;

  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*full_object, *rotated_model, rototranslations[i]);
    instances.push_back (rotated_model);
  }
*/
  /**
   * ICP
   */
/*  std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
  if (true)
  {
    cout << "--- ICP ---------" << endl;

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
      pcl::IterativeClosestPoint<PointNT, PointNT> icp;
      icp.setMaximumIterations (icp_max_iter_);
      icp.setMaxCorrespondenceDistance (icp_corr_distance_);
      icp.setInputTarget (scene);
      icp.setInputSource (instances[i]);
      pcl::PointCloud<PointType>::Ptr registered (new pcl::PointCloud<PointType>);
      icp.align (*registered);
      registered_instances.push_back (registered);
      cout << "Instance " << i << " ";
      if (icp.hasConverged ())
      {
        cout << "Aligned!" << endl;
      }
      else
      {
        cout << "Not Aligned!" << endl;
      }
    }

    cout << "-----------------" << endl << endl;
  }
*/
  /**
   * Hypothesis Verification
   */
/*  cout << "--- Hypotheses Verification ---" << endl;
  std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses

  pcl::GlobalHypothesesVerification<PointNT, PointNT> GoHv;

  GoHv.setSceneCloud (cloud_full_scene);  // Scene Cloud
  GoHv.addModels (registered_instances, true);  //Models to verify

  GoHv.setInlierThreshold (hv_inlier_th_);
  GoHv.setOcclusionThreshold (hv_occlusion_th_);
  GoHv.setRegularizer (hv_regularizer_);
  GoHv.setRadiusClutter (hv_rad_clutter_);
  GoHv.setClutterRegularizer (hv_clutter_reg_);
  GoHv.setDetectClutter (hv_detect_clutter_);
  GoHv.setRadiusNormals (hv_rad_normals_);

  GoHv.verify ();
  GoHv.getMask (hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

  for (int i = 0; i < hypotheses_mask.size (); i++)
  {
    if (hypotheses_mask[i])
    {
      cout << "Instance " << i << " is GOOD! <---" << endl;
    }
    else
    {
      cout << "Instance " << i << " is bad!" << endl;
    }
  }
  cout << "-------------------------------" << endl;
  
  */
  
  
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }
  
return true;
}


#define MAXBUFSIZE  ((int) 1e6)
Eigen::MatrixXf readMatrix(const char *filename){
    int cols = 0, rows = 0;
    float buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXf result(rows-1,cols);
    for (int i = 0; i < rows-1; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];
    return result;
}

Eigen::Matrix4f rotateZ(float radians) {
    float Sin = sinf(radians);
    float Cos = cosf(radians);

    Eigen::Matrix4f rotationMatrix( Eigen::Matrix4f::Identity() );

    rotationMatrix(0, 0) =  Cos;
    rotationMatrix(0, 1) =  Sin;
    rotationMatrix(1, 0) = -Sin;
    rotationMatrix(1, 1) =  Cos;

    return rotationMatrix;
}
Eigen::Matrix4f rotateX(float radians) {
    float Sin = sinf(radians);
    float Cos = cosf(radians);

    Eigen::Matrix4f rotationMatrix( Eigen::Matrix4f::Identity() );

    rotationMatrix(1, 1) =  Cos;
    rotationMatrix(1, 2) =  Sin;
    rotationMatrix(2, 1) = -Sin;
    rotationMatrix(2, 2) =  Cos;

    return rotationMatrix;
}
Eigen::Matrix4f rotateY(float radians) {
    float Sin = sinf(radians);
    float Cos = cosf(radians);

    Eigen::Matrix4f rotationMatrix( Eigen::Matrix4f::Identity() );

    rotationMatrix(0, 0) =  Cos;
    rotationMatrix(0, 2) =  Sin;
    rotationMatrix(2, 0) = -Sin;
    rotationMatrix(2, 2) =  Cos;

    return rotationMatrix;
}

void computePoseError(Eigen::Matrix4f object_pose, Eigen::Matrix4f gt_pose, bool ignore_z_rotation ){

  Eigen::Vector3f T_object; 
  T_object(0) = object_pose(0,3);
  T_object(1) = object_pose(1,3);
  T_object(2) = object_pose(2,3);
  
   Eigen::Vector3f T_gt; 
  T_gt(0) = gt_pose(0,3);
  T_gt(1) = gt_pose(1,3);
  T_gt(2) = gt_pose(2,3);
  
  Eigen::Vector3f T_error = T_gt - T_object;
  
  float diff_x = T_gt(0) - T_object(0), diff_y = T_gt(1) - T_object(1), diff_z = T_gt(2) - T_object(2);
  float mse = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
  float err = diff_x + diff_y + diff_z;
  std::cout << "\n================== Translation Error ====================" << std::endl;
  std::cout << T_error << std::endl;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "RMSE: " << sqrtf(mse) << std::endl;
  std::cout << "===================================================" << std::endl;
  
  Eigen::Matrix3f R_object(object_pose.block<3,3>(0,0));
  Eigen::Matrix3f R_gt(gt_pose.block<3,3>(0,0));
  Eigen::Matrix3f p0 = R_gt.transpose() * R_object;
  float trace  = p0.trace() -1; //trace
  float theta = std::acos(trace/2);
 
  std::cout << "\n================== Rotation Error ====================" << std::endl;
  std::cout << "Theta: " << theta << std::endl;
  std::cout << "Error in degree: " << theta *(180/PI) << std::endl;
  std::cout << "===================================================" << std::endl;
 
}

int main (int argc, char** argv){
  parseCommandLine (argc, argv);
   
  std::vector<int> dummy;
  std::vector<PointCloudT::Ptr > cloud_models;
  std::vector<PointCloudT::Ptr > cloud_with_normals;
  std::vector<std::pair<PointCloudT::Ptr , Eigen::Affine3f> > clouds_aligned;
  std::vector<PointCloudT::Ptr > clouds_aligned_full;
  std::vector<int> model_has_normal;
  PCDReader reader; 
  
  pcl::PCLPointCloud2 pcl2_scene_cloud;
  pcl::PCLPointCloud2 pcl2_model_cloud;
  PointCloudT::Ptr cloud_scene (new PointCloudT());
  PointCloudT::Ptr cloud_full_scene (new PointCloudT());
  PointCloudT::Ptr cloud_scene_with_normals (new PointCloudT());
  PointCloudT::Ptr cloud_model (new PointCloudT());
  PointCloudT::Ptr cloud_full_model (new PointCloudT());
   
  pcl::console::print_highlight ("Loading scene clouds...\n");
  if (reader.read (scene_filename_, pcl2_scene_cloud) != 0){
    pcl::console::print_error ("Error loading scene file!\n");
    return (1);
  }
  pcl::fromPCLPointCloud2 (pcl2_scene_cloud, *cloud_scene); 
  pcl::removeNaNFromPointCloud(*cloud_scene, *cloud_scene, dummy);
  pcl::console::print_highlight ("Finish...\n");
  
  pcl::console::print_highlight ("Loading model clouds...\n");
  
  for(unsigned int i = 0; i<model_filenames_.size(); i++){
  if (reader.read(model_filenames_.at(i), pcl2_model_cloud) != 0){
    pcl::console::print_error ("Error loading %s!\n", model_filenames_.at(i).c_str());
    return (1);
  }
  
  if(pcl::getFieldIndex(pcl2_model_cloud,"normal_x") >= 0){
      PCL_INFO("Model has normals... No normal estimation needed!!\n");
       model_has_normal.push_back(true);
    }else{
       PCL_INFO("Model has no normals... Estimating normals!!\n");
       model_has_normal.push_back(false);
    }
      
   pcl::fromPCLPointCloud2 (pcl2_model_cloud, *cloud_model); 
   pcl::removeNaNFromPointCloud(*cloud_model, *cloud_model, dummy);
   cloud_models.push_back(cloud_model);
  }
  pcl::console::print_highlight ("Finish...\n");
  /// Save full resolution scene and model clouds
  copyPointCloud(*cloud_scene, *cloud_full_scene);
  copyPointCloud(*cloud_model, *cloud_full_model);
  
  //Load ground truth pose
   Eigen::MatrixXf gt_pose;
  if(!gt_path.empty()){
   pcl::console::print_highlight ("Loading ground truth pose from %s\n", gt_path.c_str());
   gt_pose = readMatrix(gt_path.c_str());
   std::cout << "Ground truth matrix:" << std::endl;
   std::cout << gt_pose << std::endl;
  }
  
  if(remove_plane){
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
    while (cloud_scene->points.size () > 0.3 * nr_points){
      seg.setInputCloud (cloud_scene);
      seg.segment (*inliers, *coefficients);
      PCL_INFO ("Plane inliers: %u\n", inliers->indices.size ());
      if (inliers->indices.size () < 50000) break;

      extract.setInputCloud (cloud_scene);
      extract.setIndices (inliers);
      extract.filter (*cloud_scene);
    }
  }
  
  //Estimate scene normals
 
  if(use_CG_){
    //Do not downsample
    pcl::console::print_highlight ("Estimating scene normals...\n");
    subsampleAndCalculateNormals(cloud_scene, false, false);//,cloud_scene_with_normals);
  }else{
     pcl::console::print_highlight ("Downsampling and Estimating scene normals...\n");
     subsampleAndCalculateNormals(cloud_scene, true, false);//,cloud_scene_with_normals);
  }
  
  PointCloudT::Ptr cloud_model_with_normals (new PointCloudT());
  PointCloudT::Ptr aligned_cloud (new PointCloudT());
  PointCloudT::Ptr  cloud_full_trfm (new PointCloudT());
  
  for (size_t model_i = 0; model_i < cloud_models.size (); ++model_i){
    if(model_has_normal.at(model_i) == true){
      pcl::console::print_highlight ("Downsampling model...\n");
      subsample(cloud_model);//,cloud_model_with_normals);
    }else{
      pcl::console::print_highlight ("Downsampling and Estimating model normals...\n");
      subsampleAndCalculateNormals(cloud_model, true, true);
    }
    
    
    Eigen::Matrix4f alignment_transformation = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f icp_transformation = Eigen::Matrix4f::Identity();
    bool converged = false;
    if(use_prereject_){
       pcl::console::print_highlight("Estimating pose withPrerejective-RANSAC\n");
       converged = run_RANSAC(cloud_scene,cloud_model, alignment_transformation, aligned_cloud,true);
    }else if(use_RANSAC_){
       pcl::console::print_highlight("Estimating pose with RANSAC\n");
       converged = run_RANSAC(cloud_scene,cloud_model, alignment_transformation, aligned_cloud,false);
    }else if(use_PPF_){
       pcl::console::print_highlight("Estimating pose with PPF-registration\n");
      converged = run_ppf_registration(cloud_scene,cloud_model, alignment_transformation, aligned_cloud);
    }else if(use_CG_){
      pcl::console::print_highlight("Estimating pose with Corresponding grouping\n");
      converged = run_CG(cloud_scene, cloud_model,cloud_full_scene,  cloud_full_model ,alignment_transformation, aligned_cloud);
    }else{
      pcl::console::print_error("Algorithm not supported. Select: (RANSAC|Prereject|CG|PPF)");
    }
    
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    if(!flip_around_x_.empty() && !gt_path.empty()){
      
      pcl::console::print_highlight ("Loading correction matrix from %s\n", flip_around_x_.c_str());
      rotation = readMatrix(flip_around_x_.c_str());
      std::cout << "Correction matrix:" << std::endl;
      std::cout << rotation << std::endl;
    }
    //Apply correction transform
   // alignment_transformation = alignment_transformation * rotation;
    pcl::transformPointCloudWithNormals<PointNT> (*cloud_model, *aligned_cloud, alignment_transformation);
     
    if(use_icp_ && converged){
      PointCloudT::Ptr  temp (new PointCloudT());
      if(icp_pointT(aligned_cloud, cloud_scene,temp,icp_transformation)){
	 pcl::transformPointCloudWithNormals<PointNT> (*aligned_cloud, *temp, icp_transformation);
	 pcl::copyPointCloud<PointNT>(*temp, *aligned_cloud);
	 
	 pcl::transformPointCloudWithNormals<PointNT> (*cloud_full_trfm, *temp, icp_transformation);
	 pcl::copyPointCloud<PointNT>(*temp, *cloud_full_trfm);
    //  pcl::console::print_highlight("ICP converged with a rmse = %f\n", computeError(cloud_scene, aligned_cloud));
	alignment_transformation = icp_transformation * alignment_transformation ;
      }
    }

    alignment_transformation = alignment_transformation * rotation;
    
    if(gt_ignore_y_rot_ && !gt_path.empty()){
	 //Ignore the rotation around the z-axis because the object is symmitrical. 
/*	float roll, pitch, yaw = 0;
	Eigen::Matrix3f M = alignment_transformation.block<3,3>(0,0);
	Eigen::Affine3f matrix(M); 
	pcl::getEulerAngles(matrix,roll,pitch,yaw);
	
	Eigen::Quaternionf q(M);
	q.normalize ();
	std::cout << "Euler angles (Object): " << "roll: " << roll*(180/PI) << " pitch: " << pitch*(180/PI) << " yaw: " << yaw*(180/PI) << std::endl; 
	std::cout << "Quaternion angles (Object): " << "x: " << q.x() << " y: " << q.y() << " z: " << q.z()<< " w: " << q.w()*(180/PI) << std::endl; 
 
	float roll_gt, pitch_gt, yaw_gt = 0;
	Eigen::Matrix3f M1 = gt_pose.block<3,3>(0,0);
	Eigen::Affine3f matrix2(M1); 
	pcl::getEulerAngles(matrix2,roll_gt,pitch_gt,yaw_gt);
	Eigen::Quaternionf q_gt(M1);
	q_gt.normalize();
	std::cout << "Euler angles (ground_truth): " << "roll: " << roll_gt*(180/PI) << " pitch: " << pitch_gt*(180/PI) << " yaw: " << yaw_gt*(180/PI) << std::endl; 
 	std::cout << "Quaternion angles (ground_truth): " << "x: " << q_gt.x() << " y: " << q_gt.y() << " z: " << q_gt.z()<< " w: " << q_gt.w()*(180/PI) << std::endl; 
	float r = q_gt.w();
	Eigen::Quaternionf q_rot1(cos(r/2),0,sin(r/2),0);

	//Eigen::Quaternionf q_flip(cos(PI/2),0,1,0);
	Eigen::Quaternionf q_res = q_rot1;
	
	Eigen::Matrix3f mat = q_res.matrix();
	Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
	t(0,0) = mat(0,0); 	t(0,1) = mat(0,1); t(0,2) = mat(0,2);  t(0,3) = 0;
	t(1,0) = mat(1,0); 	t(1,1) = mat(1,1); t(1,2) = mat(1,2); 	t(1,3) = 0;
	t(2,0) = mat(2,0);	t(2,1) = mat(2,1); t(2,2) = mat(2,2); 	t(2,3) = 0;
	t(3,0) = 0;         	t(3,1) = 0; 	   t(3,2) = 0; 	 	t(3,3) = 1;

*/	
	 Eigen::Matrix4f diff = alignment_transformation.inverse() * gt_pose;
	 Eigen::Matrix3f diff_rot = gt_pose.block<3,3>(0,0);
	 Eigen::Quaternionf R(diff_rot);
	 std::cout << "Quaternion angles (diff): " << "x: " << R.x() << " y: " << R.y() << " z: " << R.z()<< " w: " << R.w()*(180/PI) << std::endl; 
	
	 Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
	 t(0,0) = diff(0,0); 	t(0,1) = 0; t(0,2) = diff(0,2); t(0,3) = 0;
	 t(1,0) = 0; 		t(1,1) = 1; t(1,2) = 0; 	 t(1,3) = 0;
	 t(2,0) = diff(2,0);	t(2,1) = 0; t(2,2) = diff(2,2); t(2,3) = 0;
	 t(3,0) = 0;         	t(3,1) = 0; t(3,2) = 0; 	 t(3,3) = 1;
	
	alignment_transformation = alignment_transformation* t ; 
    }
    
  //  pcl::transformPointCloudWithNormals<PointNT> (*aligned_cloud, *aligned_cloud, alignment_transformation);
  //   pcl::transformPointCloudWithNormals<PointNT> (*cloud_full_model, *cloud_full_trfm, alignment_transformation);
     
    
    
     Eigen::Matrix4f object_pose = alignment_transformation;// * icp_transformation;
     std::cout << "\n================== Object pose ====================" << std::endl;
     std::cout << object_pose << std::endl;
     std::cout << "===================================================" << std::endl;
     
  
     std::pair<PointCloudT::Ptr, Eigen::Affine3f> pair;
     pair.first = aligned_cloud;
     Eigen::Affine3f affine(object_pose);
     pair.second = affine;
     clouds_aligned.push_back(pair);
     clouds_aligned_full.push_back(cloud_full_trfm);
    
     if(!gt_path.empty())
       computePoseError(object_pose, gt_pose, true);
    
     
  }
  
  if(save_aligned_cloud){
    for(size_t model_i = 0; model_i < clouds_aligned_full.size (); ++model_i){
      std::stringstream ss; ss << "aligned_model_" << model_i; ss << ".pcd";
      pcl::io::savePCDFile(ss.str(), *clouds_aligned_full.at(model_i));
    }
    
  }
  
  if(show_vis_){
  visualization::PCLVisualizer viewer ("Pose Estimation app - Results");
  viewer.setBackgroundColor (0, 0, 0);
  viewer.registerKeyboardCallback (keyboardEventOccurred, 0);
  viewer.addPointCloud (cloud_scene,ColorHandlerT (cloud_scene, 255.0, 0.0, 0.0), "scene");
  viewer.addCoordinateSystem((25*leaf_size_),0,0,0,"world");
  
   for(size_t model_i = 0; model_i < clouds_aligned.size (); ++model_i){
      std::stringstream ss; ss << "Model_" << model_i;
      viewer.addPointCloud (clouds_aligned.at(model_i).first,ColorHandlerT (clouds_aligned.at(model_i).first, 0.0, 255.0, 0.0), ss.str());
     
  }
  viewer.spinOnce (10);
  
   while (!viewer.wasStopped ()){
    viewer.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    
   if (normals_changed_){
          viewer.removePointCloud ("scene_normals");
	  std::stringstream ss;
	  for(size_t model_i = 0; model_i < clouds_aligned.size (); ++model_i){
	    ss << "Model_normals_" << model_i;
	    viewer.removePointCloud (ss.str());
	  }
	  
        if (show_scene_normals_){
	
          viewer.addPointCloudNormals<PointNT>(cloud_scene, 1, normals_scale_, "scene_normals");
	  normals_changed_ = false;
        }
        
        if (show_model_normals_){
	  for(size_t model_i = 0; model_i < clouds_aligned.size (); ++model_i){
	    std::stringstream ss; ss << "Model_normals_" << model_i;
	    viewer.addPointCloudNormals<PointNT>(clouds_aligned.at(model_i).first, 1, normals_scale_, ss.str());
	  }
	 normals_changed_ = false;
        }
      }
      
   if(show_ground_truth_changed_){
	viewer.removePointCloud ("ground_truth");
	if(show_ground_truth_){
	 PointCloudT::Ptr  temp (new PointCloudT());
	 pcl::transformPointCloudWithNormals<PointNT> (*cloud_model, *temp, gt_pose);
	 viewer.addPointCloud (temp,ColorHandlerT (temp, 0.0, 0.0, 255.0), "ground_truth");
	}
      }
      
     if(show_frames_changed_){
        for(size_t model_i = 0; model_i < clouds_aligned.size (); ++model_i){
	  std::stringstream ss; ss << "object_" << model_i;
	  viewer.removeCoordinateSystem(ss.str());
	  }
	viewer.removeCoordinateSystem("ground_truth");
	
	if(show_frames_ && show_ground_truth_ && !gt_path.empty()){
	   Eigen::Matrix4f gtm(gt_pose);
	  Eigen::Affine3f aff(gtm);
	 viewer.addCoordinateSystem(normals_scale_*10,aff,"ground_truth");
	}
	
	if(show_frames_){
	   for(size_t model_i = 0; model_i < clouds_aligned.size (); ++model_i){
	    std::stringstream ss; ss << "object_" << model_i;
	    viewer.addCoordinateSystem(normals_scale_*10,clouds_aligned.at(model_i).second,ss.str());
	  }
	}
	
    }
  }
  
  }
  return 0;
}
