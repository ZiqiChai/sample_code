#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/console/time.h>

#include <Eigen/Dense>

using namespace pcl;
using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

typedef pcl::FPFHSignature33 FeatureT;
//typedef pcl::SHOT352 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
//typedef pcl::SHOTEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;


/// Control parameters 
bool show_vis_ (false);
bool show_frames_ (false);
bool show_scene_normals_ (false);
bool show_model_normals_ (false);
bool downsample_(false);
bool normals_changed_(false);
bool frame_changed_(false);
float normals_scale_ = 0.01;

///Common parameters
float leaf_size_ (0.005f);
float max_correspondence_distance_ = 1.0;
int nr_iterations_ =  500;
float min_sample_distance_ = leaf_size_;

std::vector<std::string> model_filenames_;
std::string scene_filename_;
std::string model_filename_;

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

        show_frames_ = !show_frames_;
	frame_changed_ = true;
        break;
   
      default:
        break;
    }
}


void showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             sac_initial_alignment - Usage Guide              	     *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " <model_filename .pcd> <scene_filename.pcd> [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:					Show this help." << std::endl;
  std::cout << "     --vis:					Show visualization" << std::endl;
  std::cout << "     --downsample:				Downsample cloud" << std::endl;
  std::cout << "     --leaf_size:				Voxel_grid leaf_size (default 0.005)" << std::endl;
  std::cout << "     --iterations:				Max iterations (default 500)" << std::endl;
  std::cout << "     --max_corr_dist:				Max correspondence distance allowed (default 1.0)" << std::endl;
  std::cout << "     --min_sample_dist:			SAC minimum sample distance (default: leaf_size)" << std::endl;
  
  std::cout << "***************************************************************************" << std::endl << std::endl;
  
 
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

  scene_filename_ = argv[filenames[1]];
  model_filename_ = argv[filenames[0]];  
  

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "--vis"))
  {
    show_vis_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "--downsample"))
  {
    downsample_ = true;
  }
 

  
  //General parameters
  pcl::console::parse_argument (argc, argv, "--leaf_size", leaf_size_);
  pcl::console::parse_argument (argc, argv, "--max_corr_dist", max_correspondence_distance_);
  pcl::console::parse_argument (argc, argv, "--min_sample_dist", min_sample_distance_);
  pcl::console::parse_argument (argc, argv, "--iterations", nr_iterations_);

}
bool my_sort(std::pair<float,Eigen::Matrix4f> i,std::pair<float,Eigen::Matrix4f> j) { return (i.first < j.first); }

void radiusOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &src_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target_cloud, double radius, int min_neighbpr_pts){
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

	if(src_cloud->size() > 0)
	{
	    try{
	      // Create the filtering object
	      pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> ror;
	      ror.setInputCloud(src_cloud);
	      ror.setRadiusSearch(radius);
	      ror.setMinNeighborsInRadius(min_neighbpr_pts);
	      ror.filter (*cloud_filtered);
	     // ror.setKeepOrganized(true);

	      pcl::copyPointCloud(*cloud_filtered, *target_cloud);
	    }catch(...)
	    {
	      PCL_ERROR("Somthing went wrong in object_modeller::radiusOutlierRemoval()");
	    }
	}
}

void statisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &src_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target_cloud, int mean)
{
	 pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud (src_cloud);
	sor.setMeanK (mean);
	sor.setStddevMulThresh (1.0);

	sor.filter (*cloud_filtered);

  	pcl::copyPointCloud(*cloud_filtered, *target_cloud);

}

void MLSApproximation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, PointCloudT::Ptr &target, double search_radius)
{
	 using namespace pcl::console;
	 
	 // Create a KD-Tree
	  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

	  // Output has the PointNormal type in order to store the normals calculated by MLS
	  PointCloudT::Ptr mls_points (new PointCloudT);

	  // Init object (second point type is for the normals, even if unused)
	  pcl::MovingLeastSquares<PointT, PointNT> mls;

	  mls.setComputeNormals (true);

	  // Set parameters
	  mls.setInputCloud (cloud);
	  mls.setPolynomialFit (true);
	  mls.setSearchMethod (tree);
	  mls.setSearchRadius (search_radius); //0.025
	  mls.setDilationVoxelSize(0.001);

	  mls.setPointDensity(0.0005);
  
         // mls.setPolynomialOrder(4); 
	
	  mls.setUpsamplingMethod(pcl::MovingLeastSquares<PointT, PointNT>::VOXEL_GRID_DILATION);
	 // mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::UpsamplingMethod::VOXEL_GRID_DILATION);

	  // Reconstruct
	  TicToc tt;
	  tt.tic ();
	  print_highlight("Computing smoothed point cloud using MLS algorithm....");
	  mls.process (*mls_points);
	  print_info ("[Done, "); print_value ("%g", tt.toc ()); print_info (" ms]\n");

	  pcl::copyPointCloud(*mls_points, *target);

}

void computeBoundingBox(pcl::PointCloud<PointT>::Ptr &cloud_filtered, Eigen::Vector3f &tra, Eigen::Quaternionf &rot,
    PointT &min_pt,PointT &max_pt){
   // Placeholder for the 3x3 covariance matrix at each surface patch
  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
  // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
  Eigen::Vector4f xyz_centroid;
  
  pcl::console::print_highlight ("Computing centroid...\n");
  pcl::compute3DCentroid(*cloud_filtered,xyz_centroid);
  
  pcl::console::print_highlight ("Computing bounding box...\n");
  Eigen::Matrix3f covariance; 
  // Compute the 3x3 covariance matrix
  pcl::computeCovarianceMatrixNormalized(*cloud_filtered, xyz_centroid, covariance); 
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors); 
  Eigen::Matrix3f eigDx = eigen_solver.eigenvectors(); 
  eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1)); 

    // move the points to the that reference frame 
  Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity()); 
  p2w.block<3,3>(0,0) = eigDx.transpose(); 
  p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * xyz_centroid.head<3>()); 
  pcl::PointCloud<PointT> cPoints; 
  pcl::transformPointCloud(*cloud_filtered, cPoints, p2w); 
 
  // std::cout << "centroid -> x: " << xyz_centroid[0] << " y: " << xyz_centroid[1] << " z: " <<  xyz_centroid[2] << std::endl;
 // PointT min_pt; 
 // PointT max_pt; 
  
  pcl::getMinMax3D (cPoints, min_pt, max_pt);
 // std::cout << "min_pt: \n " << min_pt << std::endl;
 // std::cout << "max_pt: \n " << max_pt << std::endl;
  Eigen::Vector3f mean_diag;
  mean_diag[0] = 0.5f*(max_pt.x + min_pt.x); 
  mean_diag[1] = 0.5f*(max_pt.y + min_pt.y); 
  mean_diag[2] = 0.5f*(max_pt.z + min_pt.z); 

  // final transform 
  //const Eigen::Quaternionf 
  rot = Eigen::Quaternionf(eigDx); 
  tra = eigDx*mean_diag + xyz_centroid.head<3>();    
  std::cout << "tra-> x: " << tra[0] << " y: " << tra[1] << " z: " << tra[2] << std::endl;

  
}

float computeError( PointCloudT::Ptr src,  PointCloudT::Ptr tar){
  
   float rmse = 0.0f;

    KdTreeFLANN<PointNT>::Ptr tree (new KdTreeFLANN<PointNT> ());
    tree->setInputCloud (tar);

    for (size_t point_i = 0; point_i < src->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (src->points[point_i].x) || !pcl_isfinite (src->points[point_i].y) || !pcl_isfinite (src->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (src->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      float dist = squaredEuclideanDistance (src->points[point_i], tar->points[point_nn_i]);
      rmse += dist;
    
   
    }
    rmse = sqrtf (rmse / static_cast<float> (src->points.size ()));
  return rmse;
  }
bool icp_pointT(PointCloudT::Ptr &src, PointCloudT::Ptr &tar, Eigen::Matrix4f &transform){
   
  Eigen::Matrix4f transform_;
  pcl::IterativeClosestPointWithNormals<PointNT,PointNT> icp;
  PointCloudT tmp;
  
  PCL_INFO("Refining pose using ICP with an inlier threshold of %f...\n", 40* leaf_size_);
  icp.setInputSource(src);
  icp.setInputTarget(tar);
  icp.setMaximumIterations(500);
  icp.setMaxCorrespondenceDistance(40* leaf_size_);
  icp.align(tmp);
  
  transform_ << icp.getFinalTransformation();

  
  if(!icp.hasConverged()) {
    PCL_ERROR("ICP failed!\n");
      return false;
    }else{
        transform << transform_;
    }

  PCL_INFO("Refining pose using ICP with an inlier threshold of %f...\n", 10* leaf_size_);
  icp.setMaximumIterations(200);
  icp.setMaxCorrespondenceDistance(10* leaf_size_);
  icp.align(tmp);
  
  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
    PCL_ERROR("ICP failed!\n");
      return false;
    }else{
       transform << transform_;
    }

  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 1 * leaf_size_);
  icp.setMaximumIterations(200);
  icp.setMaxCorrespondenceDistance(1 * leaf_size_);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();


  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }
  
   PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.1 * leaf_size_);      
  icp.setMaxCorrespondenceDistance(0.1 * leaf_size_);
  icp.setMaximumIterations(100);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();

  
  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
      transform << transform_;
  }    
  
  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.01 * leaf_size_);      
  icp.setMaxCorrespondenceDistance(0.01 * leaf_size_);
    icp.setMaximumIterations(100);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();
 
  
  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }else{
     transform << transform_;
  }
  
    pcl::console::print_info ("ICP Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (0,0), transform_ (0,1), transform_ (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transform_ (1,0), transform_ (1,1), transform_ (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (2,0), transform_ (2,1), transform_ (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transform_ (0,3), transform_ (1,3), transform_ (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("ICP Fitness Score: %f\n", icp.getFitnessScore()); 
    
    transform << transform_;

  return true;
}

bool icp(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &src, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &tar, Eigen::Matrix4f &transform){
   
  Eigen::Matrix4f transform_;
  pcl::IterativeClosestPoint<pcl::PointXYZRGBA,pcl::PointXYZRGBA> icp;
  pcl::PointCloud<pcl::PointXYZRGBA> tmp;
  
  PCL_INFO("Refining pose using ICP with an inlier threshold of %f...\n", 2* leaf_size_);
  icp.setInputSource(src);
  icp.setInputTarget(tar);

  icp.setMaxCorrespondenceDistance(2* leaf_size_);
  icp.align(tmp);
  
  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
    PCL_ERROR("ICP failed!\n");
      return false;
    }

  PCL_INFO("Refining pose using ICP with an inlier threshold of %f...\n", 1* leaf_size_);
  icp.setMaximumIterations(200);
  icp.setMaxCorrespondenceDistance(1* leaf_size_);
  icp.align(tmp);
  
  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
    PCL_ERROR("ICP failed!\n");
      return false;
    }

  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.1 * leaf_size_);
  icp.setMaximumIterations(100);
  icp.setMaxCorrespondenceDistance(0.1 * leaf_size_);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }
   PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.05 * leaf_size_);      
  icp.setMaxCorrespondenceDistance(0.05 * leaf_size_);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }
  
  PCL_INFO("Rerunning fine ICP with an inlier threshold of %f...\n", 0.01 * leaf_size_);      
  icp.setMaxCorrespondenceDistance(0.01 * leaf_size_);
  icp.align(tmp, icp.getFinalTransformation());

  transform_ << icp.getFinalTransformation();

  if(!icp.hasConverged()) {
      PCL_ERROR("Fine ICP failed!\n");
      return false;
  }
        
    pcl::console::print_info ("ICP Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (0,0), transform_ (0,1), transform_ (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transform_ (1,0), transform_ (1,1), transform_ (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform_ (2,0), transform_ (2,1), transform_ (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transform_ (0,3), transform_ (1,3), transform_ (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("ICP Fitness Score: %f\n", icp.getFitnessScore()); 
    
    transform << transform_;

  return true;
}


/*bool moveObjectFrame(PointCloudT::Ptr &src_cloud, PointCloudT::Ptr &tar_cloud){
  
 	Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*src_cloud,centroid);
	
	//  pcl::transformPointCloud(*src_cloud,*temp,mat);
		 
	pcl::PCA<PointNT> _pca; 
	PointNT projected; 
	PointNT reconstructed;
	CloudT cloudi = *src_cloud;
	CloudT finalCloud;
		 
	try{
	     //Do PCA for each point to preserve color information
	     //Add point cloud to force PCL to init_compute else a exception is thrown!!!HACK
	     _pca.setInputCloud(src_cloud);
	     int i;
	 //    #pragma omp parallel for
	     for(i = 0; i < (int)src_cloud->size(); i++)     {
	       _pca.project(cloudi[i],projected);
	       _pca.reconstruct (projected, reconstructed);

	       pcl::PCLPointCloud2 c;
	       pcl::toPCLPointCloud2(cloudi,c);
	       if(pcl::getFieldIndex(c,"rgba") >= 0){
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


	   pcl::io::savePCDFile("/home/thso/pca_cloud.pcd",finalCloud);	
	pcl::copyPointCloud(finalCloud,*tar_cloud);
  
}
*/

int main (int argc, char** argv){
  parseCommandLine (argc, argv);
   
  std::vector<int> dummy;
  bool model_has_normal =  false;
  bool scene_has_normal =  false;
  PCDReader reader; 
  
  pcl::PCLPointCloud2 pcl2_scene_cloud;
  pcl::PCLPointCloud2 pcl2_model_cloud;
  PointCloudT::Ptr cloud_scene (new PointCloudT());
  PointCloudT::Ptr cloud_model (new PointCloudT());
   PointCloudT::Ptr final_model (new PointCloudT());
  
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_result (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_full_model (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_full_scene (new pcl::PointCloud<pcl::PointXYZRGBA>);
 
  pcl::console::print_highlight ("Loading scene clouds...\n");
  if (reader.read (scene_filename_, pcl2_scene_cloud) != 0){
    pcl::console::print_error ("Error loading %s!\n", scene_filename_.c_str());
    return (1);
  }
  pcl::fromPCLPointCloud2 (pcl2_scene_cloud, *cloud_scene); 
  pcl::removeNaNFromPointCloud(*cloud_scene, *cloud_scene, dummy);
  pcl::console::print_highlight ("Finish...\n");
  
  pcl::console::print_highlight ("Loading model clouds...\n");
  if (reader.read(model_filename_, pcl2_model_cloud) != 0){
    pcl::console::print_error ("Error loading %s!\n", model_filename_.c_str());
    return (1);
  }
  
  if(pcl::getFieldIndex(pcl2_model_cloud,"normal_x") >= 0){
      PCL_INFO("Model has normals... No normal estimation needed!!\n");
       model_has_normal = true;
    }
      
   if(pcl::getFieldIndex(pcl2_scene_cloud,"normal_x") >= 0){
      PCL_INFO("Scene has normals... No normal estimation needed!!\n");
       scene_has_normal = true;
    }
   pcl::fromPCLPointCloud2 (pcl2_model_cloud, *cloud_model); 
   pcl::removeNaNFromPointCloud(*cloud_model, *cloud_model, dummy);
   pcl::console::print_highlight ("Finish...\n");
   
   pcl::copyPointCloud(*cloud_model, *cloud_full_model);
   pcl::copyPointCloud(*cloud_scene, *cloud_full_scene);

   pcl::io::savePCDFile("cloud_full_model.pcd", *cloud_full_model);    
   pcl::io::savePCDFile("cloud_full_scene.pcd", *cloud_full_scene);   
    
   pcl::VoxelGrid<PointNT> grid;
   if (downsample_){
     grid.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
     grid.setInputCloud (cloud_scene);
     grid.filter (*cloud_scene);
	    
     grid.setInputCloud (cloud_model);
     grid.filter (*cloud_model);
     pcl::console::print_highlight("Filtered model cloud contains %d data points\n", cloud_model->size ());
     pcl::console::print_highlight("Filtered scene cloud contains %d data points\n", cloud_scene->size ());
  }
   
  NormalEstimationOMP<PointNT, PointNT> ne;

  search::KdTree<PointNT>::Ptr search_tree (new search::KdTree<PointNT>);
  ne.setSearchMethod (search_tree);
  ne.setRadiusSearch (1.5f * leaf_size_);

  if(!scene_has_normal){
    pcl::console::print_highlight("Estimating scene normals....");
    ne.setInputCloud (cloud_scene);
    ne.compute (*cloud_scene);
    pcl::console::print_highlight ("Finish...\n");
  }
   if(!model_has_normal){
    pcl::console::print_highlight("Estimating model normals....");
    ne.setInputCloud (cloud_model);
    ne.compute (*cloud_model);
    pcl::console::print_highlight ("Finish...\n");
  }
  
//statisticalOutlierRemoval(cloud_full_model,cloud_full_model,5);
//   PointCloudT::Ptr mls (new PointCloudT());
//   MLSApproximation(cloud_full_model,mls,0.0075);
//   pcl::io::savePCDFile("mls_model.pcd", *mls);
  
   //First try an icp
 // Eigen::Matrix4f icp_transformation = Eigen::Matrix4f::Identity();
 //   icp_pointT(cloud_model, cloud_scene,icp_transformation);
 //     pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,icp_transformation);
 /* 
  radiusOutlierRemoval(cloud_full_model, cloud_full_model, 0.01, 200);
  
  //Compute bounding_bob
  Eigen::Quaternionf q_model; 
  Eigen::Vector3f t_model;
  PointT min_pt_model; 
  PointT max_pt_model; 
  computeBoundingBox(cloud_full_model,t_model,q_model,min_pt_model,max_pt_model);
  
  std::cout << "t_model-> x: " << t_model[0] << " y: " << t_model[1] << " z: " << t_model[2] << std::endl;

  radiusOutlierRemoval(cloud_full_scene, cloud_full_scene, 0.01, 200);
  
  Eigen::Quaternionf q_scene; 
  Eigen::Vector3f t_scene;
  PointT min_pt_scene; 
  PointT max_pt_scene; 
  computeBoundingBox(cloud_full_scene,t_scene,q_scene,min_pt_scene, max_pt_scene);
  
  std::cout << "t_scene-> x: " << t_scene[0] << " y: " << t_scene[1] << " z: " << t_scene[2] << std::endl;
 

  Eigen::Affine3f bb_model_transformation = Eigen::Affine3f::Identity();
  bb_model_transformation.translation() << t_model[0], t_model[1], t_model[2];
  bb_model_transformation.rotate (q_model.toRotationMatrix());
  
  Eigen::Affine3f bb_scene_transformation = Eigen::Affine3f::Identity();
  bb_scene_transformation.translation() << t_scene[0], t_scene[1], t_scene[2];
  bb_scene_transformation.rotate (q_scene.toRotationMatrix());
//  std::cout << bb_transformation << std::endl;
 */
/*  Eigen::Affine3f bb_alignment = bb_scene_transformation * bb_model_transformation.inverse();
  pcl::transformPointCloud(*cloud_full_model,*cloud_full_model,bb_alignment );
  pcl::transformPointCloudWithNormals<PointNT>(*cloud_model,*cloud_model,bb_alignment );
   float rmse;
   rmse= computeError(cloud_model, cloud_scene);
   pcl::console::print_info ("rmse: %f\n", rmse);
   pcl::console::print_info ("rotate\n");
   Eigen::Matrix4f rotation = rotateY(3.14);
   pcl::transformPointCloudWithNormals<PointNT>(*cloud_model,*cloud_model,rotation );
    rmse= computeError(cloud_model, cloud_scene);
   pcl::console::print_info ("rmse: %f\n", rmse);
  */ 
 
 //
  
//  pcl::IterativeClosestPoint<PointNT,PointNT> icp;  
//  icp.setInputTarget(cloud_scene);
//  icp.setMaximumIterations(nr_iterations_);
//  icp.setMaxCorrespondenceDistance(2* leaf_size_);
 /*
    PointCloudT::Ptr rotated_model (new PointCloudT);
    copyPointCloud(*cloud_model, *rotated_model);
    std::vector<std::pair<float,Eigen::Matrix4f> > votes;
    Eigen::Matrix4f rotation;
    for(int i = 0; i<36;i++){
      if(i < 4){
	rotation = rotateZ(i* 1.57);
       }else if(i >= 4 && i < 8) {
	 rotation = rotateY((i-4)* 1.57);
       }else if(i >= 8 && i < 12) {
	 rotation = rotateX((i-8)* 1.57);
       }else if(i >= 12 && i < 16) {
	Eigen::Matrix4f temp =  rotateY(1.57);
	Eigen::Matrix4f temp2 = rotateZ((i-12)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 16 && i < 20) {
	Eigen::Matrix4f temp =  rotateY(1.57);
	Eigen::Matrix4f temp2 = rotateX((i-16)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 20 && i < 24) {
	Eigen::Matrix4f temp =  rotateX(1.57);
	Eigen::Matrix4f temp2 = rotateZ((i-20)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 24 && i < 28) {
	Eigen::Matrix4f temp =  rotateX(1.57);
	Eigen::Matrix4f temp2 = rotateY((i-24)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 28 && i < 32) {
	Eigen::Matrix4f temp =  rotateZ(1.57);
	Eigen::Matrix4f temp2 = rotateY((i-28)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 32 && i < 36) {
	Eigen::Matrix4f temp =  rotateZ(1.57);
	Eigen::Matrix4f temp2 = rotateX((i-32)* 1.57);
	rotation = temp * temp2;
       }
       
       pcl::transformPointCloudWithNormals<PointNT>(*rotated_model, *rotated_model,rotation);
  //     icp.setInputSource(rotated_model);
     
       pcl::console::print_highlight("Alignment %d started!\n", i);
       float rmse = computeError(rotated_model, cloud_scene);
       
       //   PointCloudT registration_output;
     //  icp.align(registration_output);
    
  //    if (icp.hasConverged()){
	// Print results
	printf ("\n");
	
	pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
	pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
	pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
	pcl::console::print_info ("\n");
	pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", rotation (0,3), rotation (1,3), rotation (2,3));
	pcl::console::print_info ("\n");
        pcl::console::print_info ("rmse: %f\n", rmse);
	std::pair<float,Eigen::Matrix4f> res(rmse, rotation);
	votes.push_back(res);
    //  }else{
//	pcl::console::print_highlight("Alignment failed!\n");
      // return (1);
  //    }
    }
    
    std::sort(votes.begin(),votes.end(),my_sort);
    
    Eigen::Matrix4f best_transformation = votes.at(0).second;
    pcl::console::print_info ("Best Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", best_transformation (0,0), best_transformation (0,1), best_transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", best_transformation (1,0), best_transformation (1,1), best_transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", best_transformation (2,0), best_transformation (2,1), best_transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", best_transformation (0,3), best_transformation (1,3), best_transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Best Fitness Score: %f\n", votes.at(0).first); 
  
   pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, best_transformation);
  */
  
  
    if(true){
    pcl::console::print_highlight("FPFH - started\n");
    FeatureEstimationT pfh_est_src;
    pcl::search::KdTree<PointNT>::Ptr tree_pfh (new pcl::search::KdTree<PointNT>());
    pfh_est_src.setSearchMethod (tree_pfh);
    pfh_est_src.setRadiusSearch(3 * leaf_size_);
       // pfh_est_src.setSearchSurface (keypoints_src);
    pfh_est_src.setInputNormals (cloud_scene);
    pfh_est_src.setInputCloud (cloud_scene);

    pcl::PointCloud<FeatureT>::Ptr pfh_scene (new pcl::PointCloud<FeatureT>);
    pcl::console::print_highlight("FPFH - Compute scene features\n");
    pfh_est_src.compute (*pfh_scene);
    pcl::console::print_highlight("FPFH - finished\n");
    
    pfh_est_src.setInputNormals (cloud_model);
    pfh_est_src.setInputCloud (cloud_model);

    pcl::PointCloud<FeatureT>::Ptr pfh_model (new pcl::PointCloud<FeatureT>);
    pcl::console::print_highlight("FPFH - Compute model features\n");
    pfh_est_src.compute (*pfh_model);
    pcl::console::print_highlight("FPFH - finished\n");
    
    
   // pcl::registration::FPCSInitialAlignment<PointNT,PointNT,PointNT> fpcs;
   // fpcs.setTargetIndices(pfh_scene);
    
    pcl::SampleConsensusInitialAlignment<PointNT, PointNT, FeatureT> sac_ia_;
    // Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm

    sac_ia_.setMinSampleDistance (min_sample_distance_);
    sac_ia_.setMaxCorrespondenceDistance (max_correspondence_distance_);
    sac_ia_.setMaximumIterations (nr_iterations_);
    sac_ia_.setCorrespondenceRandomness(5);
    
    PointCloudT::Ptr rotated_model (new PointCloudT);
    
  
    copyPointCloud(*cloud_model, *rotated_model);
 /*    std::vector<std::pair<float,Eigen::Matrix4f> > votes;
    Eigen::Matrix4f rotation;
    for(int i = 0; i<36;i++){
      if(i < 4){
	rotation = rotateZ(i* 1.57);
       }else if(i >= 4 && i < 8) {
	 rotation = rotateY((i-4)* 1.57);
       }else if(i >= 8 && i < 12) {
	 rotation = rotateX((i-8)* 1.57);
       }else if(i >= 12 && i < 16) {
	Eigen::Matrix4f temp =  rotateY(1.57);
	Eigen::Matrix4f temp2 = rotateZ((i-12)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 16 && i < 20) {
	Eigen::Matrix4f temp =  rotateY(1.57);
	Eigen::Matrix4f temp2 = rotateX((i-16)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 20 && i < 24) {
	Eigen::Matrix4f temp =  rotateX(1.57);
	Eigen::Matrix4f temp2 = rotateZ((i-20)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 24 && i < 28) {
	Eigen::Matrix4f temp =  rotateX(1.57);
	Eigen::Matrix4f temp2 = rotateY((i-24)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 28 && i < 32) {
	Eigen::Matrix4f temp =  rotateZ(1.57);
	Eigen::Matrix4f temp2 = rotateY((i-28)* 1.57);
	rotation = temp * temp2;
       }else if(i >= 32 && i < 36) {
	Eigen::Matrix4f temp =  rotateZ(1.57);
	Eigen::Matrix4f temp2 = rotateX((i-32)* 1.57);
	rotation = temp * temp2;
       }
       
       pcl::transformPointCloudWithNormals<PointNT>(*rotated_model, *rotated_model,rotation);
      */
      sac_ia_.setInputSource(rotated_model);//setInputCloud (src);
      sac_ia_.setSourceFeatures (pfh_model);
      sac_ia_.setInputTarget (cloud_scene);
      sac_ia_.setTargetFeatures (pfh_scene);
      
      //pcl::console::print_highlight("Alignment %d started!\n", i);
      pcl::console::print_highlight("Alignment started!\n");
       
      PointCloudT registration_output;
      sac_ia_.align (registration_output);
      
      float rmse= computeError(registration_output.makeShared(), cloud_scene);
      pcl::console::print_info ("rmse: %f\n", rmse);
    
      if (sac_ia_.hasConverged ()){
	// Print results
	printf ("\n");
	Eigen::Matrix4f transformation = sac_ia_.getFinalTransformation ();
	pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
	pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
	pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
	pcl::console::print_info ("\n");
	pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
	pcl::console::print_info ("\n");
	pcl::console::print_info ("SAC-ia Fitness Score: %f\n", sac_ia_.getFitnessScore()); 
	std::pair<float,Eigen::Matrix4f> res(sac_ia_.getFitnessScore(), transformation);
	//votes.push_back(res);
	 pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,transformation);
	  pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, transformation);
      }else{
	pcl::console::print_highlight("Alignment failed!\n");
      // return (1);
      }
 //   }
    
     pcl::io::savePCDFile("SAC-IA_reg_src.pcd", *cloud_full_model); 
/*    std::sort(votes.begin(),votes.end(),my_sort);
    
    Eigen::Matrix4f best_transformation = votes.at(0).second;
    pcl::console::print_info ("Best Transformation:\n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", best_transformation (0,0), best_transformation (0,1), best_transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", best_transformation (1,0), best_transformation (1,1), best_transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", best_transformation (2,0), best_transformation (2,1), best_transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", best_transformation (0,3), best_transformation (1,3), best_transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Best Fitness Score: %f\n", votes.at(0).first); 
    
  // pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, best_transformation);
    pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,best_transformation);
 //    pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, best_transformation);
    icp_pointT(cloud_model, cloud_scene,icp_transformation);
    pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,icp_transformation);
  */  
    }
   
  //  pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, icp_transformation);
  //  pcl::io::savePCDFile("cloud_full_model_reg.pcd", *cloud_full_model);    
  // pcl::io::savePCDFile("cloud_full_scene_reg.pcd", *cloud_full_scene);  
    
 //   *cloud_result += *cloud_full_model; 
 //   *cloud_result += *cloud_full_scene;
    
 /*   pcl::removeNaNFromPointCloud(*cloud_result, *cloud_result, dummy);
     pcl::io::savePCDFile("raw_model.pcd", *cloud_result);   
     radiusOutlierRemoval(cloud_result, cloud_result, 0.003, 8);
     MLSApproximation(cloud_result, final_model, 0.005);
     pcl::removeNaNFromPointCloud(*final_model, *final_model, dummy);
      pcl::copyPointCloud(*final_model,*cloud_result);
      pcl::io::savePCDFile("final_model.pcd", *cloud_result);
   */   
 /*    cloud_xyzrgb->width = cloud_xyzrgb->points.size (); 
     cloud_xyzrgb->height = 1; 
 
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGBA>);
 //   pcl::copyPointCloud(*cloud_result,*cloud_xyzrgb);
    
   //   pcl::io::savePCDFileASCII("test_model.pcd", *cloud_xyzrgb);

 /*   cloud_xyzrgb->points.resize(cloud_result->size());
    for (size_t i = 0; i < cloud_result->points.size(); i++) {
    cloud_xyzrgb->points[i].x = cloud_result->points[i].x;
    cloud_xyzrgb->points[i].y = cloud_result->points[i].y;
    cloud_xyzrgb->points[i].z = cloud_result->points[i].z;
    
    cloud_xyzrgb->points[i].rgba = cloud_result->points[i].rgba;
    }
    */
   

  if(show_vis_){
  visualization::PCLVisualizer viewer ("SAC Initial Alignment app - Results");
  viewer.setBackgroundColor (0, 0, 0);
  viewer.registerKeyboardCallback (keyboardEventOccurred, 0);
 viewer.addPointCloud (cloud_scene,ColorHandlerT (cloud_scene, 255.0, 0.0, 0.0), "scene");
  viewer.addPointCloud (cloud_model,ColorHandlerT (cloud_model, 0.0, 255.0, 0.0), "model");
//  viewer.addPointCloud (cloud_model,ColorHandlerT (cloud_model, 0.0, 255.0, 0.0), "model");
 // viewer.addPointCloud (cloud_full_model,pcl::visualization::PointCloudColorHandlerRGBAField<pcl::PointXYZRGBA> (cloud_full_model), "result1");
//  viewer.addCube(t_model, q_model, max_pt_model.x - min_pt_model.x, max_pt_model.y - min_pt_model.y, max_pt_model.z - min_pt_model.z,"bb_model"); 
//  viewer.addCube(t_scene, q_scene, max_pt_scene.x - min_pt_scene.x, max_pt_scene.y - min_pt_scene.y, max_pt_scene.z - min_pt_scene.z,"bb_scene"); 
//  viewer.addCoordinateSystem(0.2,bb_model_transformation,"model_center",0);
 //  viewer.addCoordinateSystem(0.2,bb_scene_transformation,"scene_center",0);
   
 // viewer.addPointCloud (cloud_full_scene,pcl::visualization::PointCloudColorHandlerRGBAField<pcl::PointXYZRGBA> (cloud_full_scene), "result");
  viewer.spinOnce (10);
  
   while (!viewer.wasStopped ()){
    viewer.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    
   if (normals_changed_){
          viewer.removePointCloud ("scene_normals");
	  viewer.removePointCloud ("model_normals");
	 
        if (show_scene_normals_){
	
          viewer.addPointCloudNormals<PointNT>(cloud_scene, 1, normals_scale_, "scene_normals");
	  normals_changed_ = false;
        }
        
        if (show_model_normals_){
	  viewer.addPointCloudNormals<PointNT>(cloud_model, 1, normals_scale_,"model_normals");
	  normals_changed_ = false;
        }
      }
      
      if(frame_changed_){
	viewer.removeAllCoordinateSystems();
	if(show_frames_){
	 viewer.addCoordinateSystem(normals_scale_,0,0,0,"world_frame");
	frame_changed_= false;
	}
      }
  }
  
  }
  
 
  return 0;
}
