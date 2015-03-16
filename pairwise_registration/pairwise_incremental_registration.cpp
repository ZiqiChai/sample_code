/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

/* \author Radu Bogdan Rusu
 * adaptation Raphael Favier*/

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <covis/feature/normal_correction_manifold.h>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;


typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNormalT,PointNormalT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

// This is a tutorial so we can afford having global variables 
	//our visualizer
	pcl::visualization::PCLVisualizer *p;
	//its left and right viewports
	int vp_1, vp_2;

//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
 *
 */
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);

  PCL_INFO ("Press q to begin the registration.\n");
  p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source");
  p->removePointCloud ("target");


  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");


  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Load a set of PCD files that we want to register together
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd");
  // Suppose the first argument is the actual test model
  for (int i = 1; i < argc; i++)
  {
    std::string fname = std::string (argv[i]);
    // Needs to be at least 5: .plot
    if (fname.size () <= extension.size ())
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower);

    //check that the argument is a pcd file
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // Load the cloud and saves it into the global list of models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);

      models.push_back (m);
    }
  }
}

void computeBoundingBox( PointCloud::Ptr &cloud_filtered, Eigen::Vector3f &tra, Eigen::Quaternionf &rot,
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
////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.002, 0.002, 0.002);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);
  
  covis::feature::NormalCorrectionManifold<PointNormalT> ncm;
  ncm.setK(350);
 // ncm.compute(*cloud);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  ncm.compute(*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
    ncm.compute(*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-9);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.05);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);



  //
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (5);
  for (int i = 0; i < 300; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);
    
    if(!reg.hasConverged())
      break;

		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
  }

  if(reg.hasConverged()){
  std::cout << "icp converged with a fitness score on " << reg.getFitnessScore() <<std::endl; 
  }
 
 /*PCL_INFO ("Running fine registration.\n");
   reg.setMaxCorrespondenceDistance (0.001); 
   reg.setInputSource(points_with_normals_src);
   reg.align (*reg_result);
   points_with_normals_src = reg_result;
   */
   showCloudsRight(points_with_normals_tgt, points_with_normals_src);
	//
  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  p->removePointCloud ("source");
  p->removePointCloud ("target");

  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

	PCL_INFO ("Press q to continue the registration.\n");
  p->spin ();

  p->removePointCloud ("source"); 
  p->removePointCloud ("target");

  //add the source to the transformed target
  *output += *cloud_src;
  
  final_transform = targetToSource;
 }

 
void sac_ia(PointCloudWithNormals::Ptr &src, PointCloudWithNormals::Ptr &tar){
  
    pcl::console::print_highlight("FPFH - started\n");
    FeatureEstimationT pfh_est_src;
    pcl::search::KdTree<PointNormalT>::Ptr tree_pfh (new pcl::search::KdTree<PointNormalT>());
    pfh_est_src.setSearchMethod (tree_pfh);
    pfh_est_src.setRadiusSearch(0.001);
       // pfh_est_src.setSearchSurface (keypoints_src);
    pfh_est_src.setInputNormals (src);
    pfh_est_src.setInputCloud (src);

    pcl::PointCloud<FeatureT>::Ptr pfh_scene (new pcl::PointCloud<FeatureT>);
    pcl::console::print_highlight("FPFH - Compute scene features\n");
    pfh_est_src.compute (*pfh_scene);
    pcl::console::print_highlight("FPFH - finished\n");
    
    pfh_est_src.setInputNormals (tar);
    pfh_est_src.setInputCloud (tar);

    pcl::PointCloud<FeatureT>::Ptr pfh_model (new pcl::PointCloud<FeatureT>);
    pcl::console::print_highlight("FPFH - Compute model features\n");
    pfh_est_src.compute (*pfh_model);
    pcl::console::print_highlight("FPFH - finished\n");
    
/*    
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
      
      sac_ia_.setInputSource(rotated_model);//setInputCloud (src);
      sac_ia_.setSourceFeatures (pfh_model);
      sac_ia_.setInputTarget (cloud_scene);
      sac_ia_.setTargetFeatures (pfh_scene);
      
      pcl::console::print_highlight("Alignment %d started!\n", i);
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
	votes.push_back(res);
      }else{
	pcl::console::print_highlight("Alignment failed!\n");
      // return (1);
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
    
  // pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, best_transformation);
    pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,best_transformation);
 //    pcl::transformPointCloud(*cloud_full_model, *cloud_full_model, best_transformation);

    pcl::transformPointCloudWithNormals<PointNT>(*cloud_model, *cloud_model,icp_transformation);
     
  */
}

/* ---[ */
int main (int argc, char** argv)
{
  // Load data
  std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  loadData (argc, argv, data);

  // Check user input
  if (data.empty ())
  {
    PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]);
    PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc");
    return (-1);
  }
  PCL_INFO ("Loaded %d datasets.", (int)data.size ());
  
  // Create a PCLVisualizer object
  p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
  p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);

	PointCloud::Ptr result (new PointCloud), source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  
  Eigen::Quaternionf q_scene; 
  Eigen::Vector3f t_scene;
  Eigen::Quaternionf q_model; 
  Eigen::Vector3f t_model;
  
  PointT min_pt_model; 
  PointT max_pt_model; 
  PointT min_pt_scene; 
  PointT max_pt_scene; 
  
  for (size_t i = 1; i < data.size (); ++i)
  {
    source = data[i-1].cloud;
    target = data[i].cloud;
    
  computeBoundingBox(target,t_scene,q_scene,min_pt_scene, max_pt_scene);
  computeBoundingBox(source,t_model,q_model,min_pt_model,max_pt_model);
  
  Eigen::Affine3f bb_model_transformation = Eigen::Affine3f::Identity();
  bb_model_transformation.translation() << t_model[0], t_model[1], t_model[2];
  bb_model_transformation.rotate (Eigen::Matrix3f::Identity());
  
  Eigen::Affine3f bb_scene_transformation = Eigen::Affine3f::Identity();
  bb_scene_transformation.translation() << t_scene[0], t_scene[1], t_scene[2];
  bb_scene_transformation.rotate (Eigen::Matrix3f::Identity());

  Eigen::Affine3f bb_alignment = bb_scene_transformation * bb_model_transformation.inverse();
  //pcl::transformPointCloud(*source,*source,bb_alignment );

  
    // Add visualization data
    showCloudsLeft(source, target);

    PointCloud::Ptr temp (new PointCloud);
    PCL_INFO ("Aligning %s (%d) with %s (%d).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());
    pairAlign (source, target, temp, pairTransform, true);
    

    //transform current pair into the global transform
    pcl::transformPointCloud (*temp, *result, GlobalTransform);

    //update the global transform
    GlobalTransform = GlobalTransform * pairTransform;

		//save aligned pair, transformed into the first cloud's frame
    std::stringstream ss;
    ss << i << ".pcd";
    pcl::io::savePCDFile (ss.str (), *result, true);

  }
}
/* ]--- */