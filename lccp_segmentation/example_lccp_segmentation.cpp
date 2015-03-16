/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 */

// Stdlib
#include <stdlib.h>
#include <cmath>
#include <limits.h>

#include <boost/format.hpp>


// PCL input/output
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

//PCL other
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// The segmentation class this example is for
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/normal_refinement.h>

// VTK
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPolyLine.h>

/// *****  Type Definitions ***** ///

typedef pcl::PointXYZRGBA PointT;  // The point type used for input
typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;
typedef pcl::PointCloud<PointT> CloudT;

/// Callback and variables

bool show_normals = false, normals_changed = false;
bool show_adjacency = false;
bool show_supervoxels = false;
bool show_help = true;
float normals_scale;

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
      case (int) '1':
        show_normals = !show_normals;
        normals_changed = true;
        break;
      case (int) '2':
        show_adjacency = !show_adjacency;
        break;
      case (int) '3':
        show_supervoxels = !show_supervoxels;
        break;
      case (int) '4':
        normals_scale *= 1.25;
        normals_changed = true;
        break;
      case (int) '5':
        normals_scale *= 0.8;
        normals_changed = true;
        break;
      case (int) 'd':
      case (int) 'D':
        show_help = !show_help;
        break;
      default:
        break;
    }
}

/// *****  Prototypes helper functions***** ///

/** \brief Displays info text in the specified PCLVisualizer
 *  \param[in] viewer_arg The PCLVisualizer to modify  */
void
printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg);

/** \brief Removes info text in the specified PCLVisualizer
 *  \param[in] viewer_arg The PCLVisualizer to modify  */
void
removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg);

void toSpherical(pcl::PointNormal normal, float &radius, float &theta, float &phi ){
  
  radius = std::sqrt((normal.normal_x*normal.normal_x) + (normal.normal_y*normal.normal_y) + (normal.normal_z*normal.normal_z));
  theta = std::acos((normal.normal_z/radius));
  phi = std::atan(normal.normal_y/normal.normal_x);
  
}

void label_cloud(CloudT::Ptr &src_cloud, pcl::PointCloud<pcl::PointXYZL>::Ptr &label_cloud, uint32_t label){
  
   CloudT::iterator cloud_iter;
   for (cloud_iter = src_cloud->begin(); cloud_iter != src_cloud->end(); cloud_iter++) {
	pcl::PointXYZL p;
	p.x = cloud_iter->x; p.y = cloud_iter->y; p.z = cloud_iter->z;
	p.label = label;
	label_cloud->points.push_back(p);
   } 
}

void computePlaneCoeff(CloudT::Ptr &src_cloud,pcl::ModelCoefficients::Ptr coefficients, double dist_threads)
{
   //*********************************************************************//
   //	Plane fitting
   /**********************************************************************/
    
  // pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

   // Create the segmentation object
   pcl::SACSegmentation<PointT> seg;
   // Optional
   seg.setOptimizeCoefficients (true);
   // Mandatory
   seg.setModelType (pcl::SACMODEL_PLANE);
   seg.setMethodType (pcl::SAC_RANSAC);
   seg.setDistanceThreshold (dist_threads);

   seg.setInputCloud (src_cloud);
   seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
   {
     PCL_ERROR ("Could not estimate a planar model for the given dataset.");
     //return (-1);
   }

   std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                       << coefficients->values[1] << " "
                                       << coefficients->values[2] << " "
                                       << coefficients->values[3] << std::endl;



}

void computePrincipalCurvature(CloudT::Ptr &src_cloud){
 
  // Compute the normals
  pcl::NormalEstimationOMP<PointT, pcl::Normal> normal_estimation;
  normal_estimation.setInputCloud (src_cloud);

  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  normal_estimation.setSearchMethod (tree);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);

  normal_estimation.setRadiusSearch (0.03);

  normal_estimation.compute (*cloud_with_normals);

  // Setup the principal curvatures computation
  pcl::PrincipalCurvaturesEstimation<PointT, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

  // Provide the original point cloud (without normals)
  principal_curvatures_estimation.setInputCloud (src_cloud);

  // Provide the point cloud with normals
  principal_curvatures_estimation.setInputNormals (cloud_with_normals);

  // Use the same KdTree from the normal estimation
  principal_curvatures_estimation.setSearchMethod (tree);
  principal_curvatures_estimation.setRadiusSearch (0.05);

  // Actually compute the principal curvatures
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
  principal_curvatures_estimation.compute (*principal_curvatures);

  std::cout << "output points.size (): " << principal_curvatures->points.size () << std::endl;

  // Display and retrieve the shape context descriptor vector for the 0th point.
  pcl::PrincipalCurvatures descriptor = principal_curvatures->points[0];
  std::cout << descriptor << std::endl;

}

void my_flipNormalTowardsViewpoint (pcl::Normal &point, float vp_x, float vp_y, float vp_z,
                              float &nx, float &ny, float &nz)
  {
    // See if we need to flip any plane normals
    vp_x -= point.normal_x;
    vp_y -= point.normal_y;
    vp_z -= point.normal_z;

    // Dot product between the (viewpoint - point) and the plane normal
    float cos_theta = (vp_x * nx + vp_y * ny + vp_z * nz);

    // Flip the plane normal
    if (cos_theta < 0)
    {
      nx *= -1;
      ny *= -1;
      nz *= -1;
    }
  }

/// ---- main ---- ///
int
main (int argc,
      char ** argv)
{
  if (argc < 2)  /// Print Info
  {
    pcl::console::print_info (
\
        "\n\
-- pcl::LCCPSegmentation example -- :\n\
\n\
Syntax: %s input.pcd  [Options] \n\
\n\
Output:\n\
  -o <outname> \n\
          Write segmented point cloud to disk (Type XYZL). If this option is specified without giving a name, the <outputname> defaults to <inputfilename>_out.pcd.\n\
          The content of the file can be changed with the -add and -bin flags\n\
  -novis  Disable visualization\n\
Output options:\n\
  -add    Instead of XYZL, append a label field to the input point cloud which holds the segmentation results (<input_cloud_type>+L)\n\
          If a label field already exists in the input point cloud it will be overwritten by the segmentation\n\
  -bin    Save a binary pcd-file instead of an ascii file \n\
  -so     Additionally write the colored supervoxel image to <outfilename>_svcloud.pcd\n\
  \n\
Supervoxel Parameters: \n\
  -v <voxel resolution> \n\
  -s <seed resolution> \n\
  -c <color weight> \n\
  -z <spatial weight> \n\
  -n <normal_weight> \n\
  -tvoxel - Use single-camera-transform for voxels (Depth-Dependent-Voxel-Grid)\n\
  -refine - Use supervoxel refinement\n\
  -nonormals - Ignore the normals from the input pcd file\n\
  \n\
LCCPSegmentation Parameters: \n\
  -ct <concavity tolerance angle> - Angle threshold for concave edges to be treated as convex. \n\
  -st <smoothness threshold> - Invalidate steps. Value from the interval [0,1], where 0 is the strictest and 1 equals 'no smoothness check' \n\
  -ec - Use extended (less local) convexity check\n\
  -sc - Use sanity criterion to invalidate singular connected patches\n\
  -smooth <mininmal segment size>  - Remove small segments which have fewer points than minimal segment size\n\
    \n",
        argv[0]);
    return (1);
  }

  /// -----------------------------------|  Preparations  |-----------------------------------

  bool sv_output_specified = pcl::console::find_switch (argc, argv, "-so");
  bool show_visualization = (not pcl::console::find_switch (argc, argv, "-novis"));
  bool ignore_provided_normals = pcl::console::find_switch (argc, argv, "-nonormals");
  bool add_label_field = pcl::console::find_switch (argc, argv, "-add");
  bool save_binary_pcd = pcl::console::find_switch (argc, argv, "-bin");
  
  /// Create variables needed for preparations
  std::string outputname ("");
  pcl::PointCloud<PointT>::Ptr input_cloud_ptr (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr input_normals_ptr (new pcl::PointCloud<pcl::Normal>);
  bool has_normals = false;

 std::vector<CloudT,  Eigen::aligned_allocator_indirection<CloudT> > clusters;
  
  /// Get pcd path from command line
  std::string pcd_filename = argv[1];
  PCL_INFO ("Loading pointcloud\n");
  
  /// check if the provided pcd file contains normals
  pcl::PCLPointCloud2 input_pointcloud2;
  if (pcl::io::loadPCDFile (pcd_filename, input_pointcloud2))
  {
    PCL_ERROR ("ERROR: Could not read input point cloud %s.\n", pcd_filename.c_str ());
    return (3);
  }
  pcl::fromPCLPointCloud2 (input_pointcloud2, *input_cloud_ptr);
  if (!ignore_provided_normals)
  {
    if (pcl::getFieldIndex (input_pointcloud2,"normal_x") >= 0)
    {
      pcl::fromPCLPointCloud2 (input_pointcloud2, *input_normals_ptr);
      has_normals = true;

      //NOTE Supposedly there was a bug in old PCL versions that the orientation was not set correctly when recording clouds. This is just a workaround.
      if (input_normals_ptr->sensor_orientation_.w () == 0)
      {
        input_normals_ptr->sensor_orientation_.w () = 1;
        input_normals_ptr->sensor_orientation_.x () = 0;
        input_normals_ptr->sensor_orientation_.y () = 0;
        input_normals_ptr->sensor_orientation_.z () = 0;
      }
    }
    else{
       PCL_WARN ("Could not find normals in pcd file. Normals will be calculated. This only works for single-camera-view pointclouds.\n");
      has_normals = false;
       /*
       pcl::console::print_highlight ("Estimating scene normals...\n");
        
       pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
       pcl::NormalEstimationOMP<PointT,pcl::Normal> nest;
       nest.setRadiusSearch (0.02);
      // nest.setViewPoint(1,0,1);
       nest.setViewPoint(0, 0,1);
       
       
     //  std::cout << "sensor orgin: [" << input_cloud_ptr->sensor_origin_[0] << "," << input_cloud_ptr->sensor_origin_[1] << "," << input_cloud_ptr->sensor_origin_[2] << "]" << std::endl;
       //nest.setKSearch (20);  
       nest.setInputCloud (input_cloud_ptr);
       nest.compute (*input_normals_ptr);
       
  //     for (unsigned int i = 0; i < input_normals_ptr->size (); ++i){
//	my_flipNormalTowardsViewpoint (input_normals_ptr->points.at(i), 0, 1, 0,input_normals_ptr->points.at(i).normal_x, input_normals_ptr->points.at(i).normal_y, input_normals_ptr->points.at(i).normal_z);
  //    }
      
    //   pcl::NormalRefinement<pcl::Normal> nr;
    //   nr.setCorrespondences();
    //   nr.setInputCloud(input_normals_ptr);
    //   nr.filter(*input_normals_ptr);
       
       has_normals = true;
 */
  
      
    }
     // PCL_WARN ("Could not find normals in pcd file. Normals will be calculated. This only works for single-camera-view pointclouds.\n");
  }
  PCL_INFO ("Done making cloud\n");

  ///  Create outputname if not given
  bool output_specified = pcl::console::find_switch (argc, argv, "-o");
  if (output_specified)
  {
    pcl::console::parse (argc, argv, "-o", outputname);

    // If no filename is given, get output filename from inputname (strip seperators and file extension)
    if (outputname.empty () || (outputname.at (0) == '-'))
    {
      outputname = pcd_filename;
      size_t sep = outputname.find_last_of ('/');
      if (sep != std::string::npos)
        outputname = outputname.substr (sep + 1, outputname.size () - sep - 1);

      size_t dot = outputname.find_last_of ('.');
      if (dot != std::string::npos)
        outputname = outputname.substr (0, dot);
    }
  }


///Compute plane coefficients
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  computePlaneCoeff(input_cloud_ptr,coefficients,0.010);
/// -----------------------------------|  Main Computation  |-----------------------------------

  ///  Default values of parameters before parsing
  // Supervoxel Stuff
  float voxel_resolution = 0.0035f;
  float seed_resolution = 0.022f;
  float color_importance = 2.0f;
  float spatial_importance = 5.0f;
  float normal_importance = 8.0f;
  bool use_single_cam_transform = false;
  bool use_supervoxel_refinement = false;

  // LCCPSegmentation Stuff
  float concavity_tolerance_threshold = 13;
  float smoothness_threshold = 0.1;
  uint32_t min_segment_size = 10;
  bool use_extended_convexity = true;
  bool use_sanity_criterion = true;
  
  ///  Parse Arguments needed for computation
  //Supervoxel Stuff
  use_single_cam_transform = pcl::console::find_switch (argc, argv, "-tvoxel");
  use_supervoxel_refinement = pcl::console::find_switch (argc, argv, "-refine");

  pcl::console::parse (argc, argv, "-v", voxel_resolution);
  pcl::console::parse (argc, argv, "-s", seed_resolution);
  pcl::console::parse (argc, argv, "-c", color_importance);
  pcl::console::parse (argc, argv, "-z", spatial_importance);
  pcl::console::parse (argc, argv, "-n", normal_importance);

  normals_scale = seed_resolution / 2.0;
  
  // Segmentation Stuff
  pcl::console::parse (argc, argv, "-ct", concavity_tolerance_threshold);
  pcl::console::parse (argc, argv, "-st", smoothness_threshold);
  use_extended_convexity = pcl::console::find_switch (argc, argv, "-ec");
  uint k_factor = 0;
  if (use_extended_convexity)
    k_factor = 1;
  use_sanity_criterion = pcl::console::find_switch (argc, argv, "-sc");
  pcl::console::parse (argc, argv, "-smooth", min_segment_size);

  /// Preparation of Input: Supervoxel Oversegmentation

  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, use_single_cam_transform);
  super.setInputCloud (input_cloud_ptr);
  if (has_normals)
    super.setNormalCloud (input_normals_ptr);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

  PCL_INFO ("Extracting supervoxels\n");
  super.extract (supervoxel_clusters);

  if (use_supervoxel_refinement)
  {
    PCL_INFO ("Refining supervoxels\n");
    super.refineSupervoxels (2, supervoxel_clusters);
  }
  std::stringstream temp;
  temp << "  Nr. Supervoxels: " << supervoxel_clusters.size () << "\n";
  PCL_INFO (temp.str ().c_str ());

  PCL_INFO ("Getting supervoxel adjacency\n");
  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  super.getSupervoxelAdjacency (supervoxel_adjacency);

  /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
  pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);

  /// The Main Step: Perform LCCPSegmentation

  PCL_INFO ("Starting Segmentation\n");
  pcl::LCCPSegmentation<PointT> lccp;
  lccp.setConcavityToleranceThreshold (concavity_tolerance_threshold);
  lccp.setSanityCheck (use_sanity_criterion);
  lccp.setSmoothnessCheck (true, voxel_resolution, seed_resolution, smoothness_threshold);
  lccp.setKFactor (k_factor);
  lccp.segment (supervoxel_clusters, supervoxel_adjacency);

  if (min_segment_size > 0)
  {
    PCL_INFO ("Removing small segments\n");
    lccp.removeSmallSegments (min_segment_size);
  }

  PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");
  pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud ();
  pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = sv_labeled_cloud->makeShared ();
  lccp.relabelCloud (*lccp_labeled_cloud);
  pcl::io::savePCDFile("/home/thso/lccp_labeled_cloud.pcd",*lccp_labeled_cloud);
  SuperVoxelAdjacencyList sv_adjacency_list;
  lccp.getSVAdjacencyList (sv_adjacency_list);  // Needed for visualization

 // std::map<uint32_t, std::vector<uint32_t> > segment_supervoxel_map;   
//  lccp.getSegmentSupervoxelMap(segment_supervoxel_map);
//  std::cout << "Number of segments: " << segment_supervoxel_map.size() << std::endl;
/*
  std::map<uint32_t, std::vector<uint32_t> >::iterator seg_iter;
     for (seg_iter = segment_supervoxel_map.begin(); seg_iter != segment_supervoxel_map.end(); seg_iter++) {
       ///First get the label
       uint32_t segment_label = seg_iter->first;
       std::cout << "Printing VoxelID for segment: " << segment_label << std::endl;
       std::vector<uint32_t> superVoxelID = seg_iter->second;
       
       pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segment_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	  for(std::vector<uint32_t>::iterator it = superVoxelID.begin(); it!= superVoxelID.end();it++){
	      uint32_t voxelID = *it;
	      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = supervoxel_clusters.at(voxelID)->voxels_;
	      *segment_cloud += *cloud;
	  }
	std::stringstream ss;
	ss << "/home/thso/segment_" << segment_label; ss << ".pcd";
	pcl::io::savePCDFile(ss.str(),*segment_cloud);
	segment_cloud->clear();
     }

*/

  
    
  /// Creating Colored Clouds and Output
  if (lccp_labeled_cloud->size () == input_cloud_ptr->size ())
  {
   
 //  if (pcl::getFieldIndex (*src_cloud, "label") >= 0)
   //       PCL_WARN ("Input cloud already has a label field. It will be overwritten by the lccp segmentation output.\n");
  
      ///Get all LCCP segments
      std::map<uint32_t, std::vector<uint32_t> > segment_supervoxel_map;   
      lccp.getSegmentSupervoxelMap(segment_supervoxel_map);
      if(segment_supervoxel_map.empty()){ 
	 pcl::console::print_error("ERROR: Failed to get segmented supervoxel map for LCCP algorithm!");
	 return false;
      }
        
      std::map<uint32_t, pcl::PointIndices::Ptr> segment_indice;
  /*    for(uint32_t i = 0; i< _num_of_segment; i++){
	std::cout << "i: " << i << std::endl;
	 pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	 segment_indice.insert(std::pair<uint32_t, pcl::PointIndices::Ptr>(i,inliers));
      }
    */  
      ///Iterate through LCCP labled cloud to copy label to inlier map
      pcl::PointCloud<pcl::PointXYZL>::iterator cloud_iter;
      std::map<uint32_t, uint32_t> pointcloud_label_to_new_label;
      int count = 0;
      for (cloud_iter = lccp_labeled_cloud->begin(); cloud_iter != lccp_labeled_cloud->end(); cloud_iter++) {
	//TODO: Kig på om label er ny. hvis: lig gamle lable i map på 1 til antal segmenters plads
	  uint32_t label =cloud_iter->label; 
	  //if(label != 1) std::cout << label << " " << std::endl;
	  if(pointcloud_label_to_new_label.count(label) == 0){ //The label dosen't exist
	     pcl::console::print_highlight("adding original label %d as label %d in inlier map\n",label, count);
	     ///Initialize map to store inliers(segments)
	     ///(Create new inlier object to hold the segment points)
	     pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	     segment_indice.insert(std::pair<uint32_t, pcl::PointIndices::Ptr>(count,inliers));
             
	     pointcloud_label_to_new_label.insert(std::pair<uint32_t,uint32_t>(label, count));
	     count++;
	  }
	  //label already exist. get the label position from pointcloud_label_to_new_label
	   uint32_t index = pointcloud_label_to_new_label.find(label)->second;
	//   std::cout << "Save label :" << pointcloud_label_to_new_label.find(label)->first << " at index: " << index << std::endl;
	
	    //Map label to point
	    segment_indice.at(index)->indices.push_back(cloud_iter - lccp_labeled_cloud->begin());
	  
      }
      pcl::console::print_highlight("Cloud has %d LCCP segments\n", segment_indice.size());
   //   pcl::console::print_highlight("Number of labels : %d\n", segment_indice.size());
      
      CloudT::Ptr segment_cloud (new CloudT);
      pcl::PointCloud<pcl::PointXYZL>::Ptr labled_cloud (new pcl::PointCloud<pcl::PointXYZL>);
      /// Extract full resolution point cloud for each segment 
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud (input_cloud_ptr);
      
      pcl::PointIndices::Ptr inliers_sac (new pcl::PointIndices);

      // Create the segmentation object
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.015);
      
      ///Segment iterator
   //   for(uint32_t i = 0; i<= segment_indice.size()-1; i++){
	  //Get all supervoxels in one segment
	  std::map<uint32_t, pcl::PointIndices::Ptr >::iterator seg_iter;
	//  std::cout << "segment_supervoxel_map size: " << segment_supervoxel_map.size() << std::endl;
	  for(seg_iter = segment_indice.begin(); seg_iter != segment_indice.end(); seg_iter++) {
		///First get the label
		uint32_t segment_label = seg_iter->first;
		std::cout << "-------------------Segment " << segment_label << " --------------------------" << std::endl;
	/*	std::vector<uint32_t> supervoxels_in_segment = seg_iter->second;
	           pcl::console::print_highlight("Number of supervoxels in segment: %d\n ",supervoxels_in_segment.size());
		  ///Supervoxel iterator in one segment
		  float theta_avg = 0;
		  for(uint32_t j = 0; j< supervoxels_in_segment.size(); j++){
		    uint32_t id = supervoxels_in_segment.at(j);
		    pcl::PointNormal normal;
		    supervoxel_clusters.at(id)->getCentroidPointNormal(normal);
		    float radius, theta, phi = 0;
		    toSpherical(normal, radius, theta, phi); 
		    theta_avg += theta;
		    //std::cout << "Normal_x: " << normal.normal_x << " Normal_y: " << normal.normal_y << " Normal_z: " << normal.normal_z << std::endl;
		  //  std::cout << " theta: " << theta << " phi: " << phi << std::endl;
		  }
		  float samples = (float(supervoxels_in_segment.size())); 
		  */
		 // std::cout << " samples: " << samples << std::endl;
		 // std::cout << " theta_acc: " << theta_avg << std::endl;
		//  pcl::console::print_highlight(" theta_avg: %f\n", theta_avg/samples);
		//  theta_avg = theta_avg/samples;
		  
		
		  pcl::PointIndices::Ptr inliers = segment_indice.at(segment_label);
		  if (inliers->indices.size () != 0){
		      extract.setIndices (inliers);
		      extract.setNegative (false);
		      extract.filter (*segment_cloud);
		      
		      pcl::console::print_highlight("%d points in segment\n",segment_cloud->points.size());
		       ///Only consider clouds larger than 40 points
		      if(segment_cloud->points.size() > 40){ 
		        //Estimate the plane coefficients
			 pcl::ModelCoefficients::Ptr seg_coefficients (new pcl::ModelCoefficients);
			 seg.setInputCloud (segment_cloud);
			 seg.segment (*inliers_sac, *seg_coefficients);
			 
			 if (inliers_sac->indices.size () == 0)
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
	      
			 std::cerr << "Model coefficients: "  << seg_coefficients->values[0] << " "
								  << seg_coefficients->values[1] << " "
								  << seg_coefficients->values[2] << " "
								  << seg_coefficients->values[3] << std::endl;
			 //coefficients.reset();
			 pcl::PointNormal segment_normal;
			 segment_normal.normal_x = seg_coefficients->values[0];
			 segment_normal.normal_y = seg_coefficients->values[1];
			 segment_normal.normal_z = seg_coefficients->values[2];
			 
			 pcl::PointNormal plane_normal;
			 plane_normal.normal_x = coefficients->values[0];
			 plane_normal.normal_y = coefficients->values[1];
			 plane_normal.normal_z = coefficients->values[2];
			 
			 float seg_radius, seg_theta,  seg_phi = 0;
		         toSpherical(segment_normal, seg_radius, seg_theta, seg_phi); 
			 std::cout <<"Segment - radius: " << seg_radius << " theta: " << seg_theta << " phi: " << seg_phi << std::endl;
		   
			 float plane_radius, plane_theta,  plane_phi = 0;
			 toSpherical(plane_normal, plane_radius, plane_theta, plane_phi); 
			 std::cout <<"table plane - radius: " << plane_radius << " theta: " << plane_theta << " phi: " << plane_phi << std::endl;
		   
			  //Compute centroid 
			  Eigen::Vector4d centroid;
			  pcl::compute3DCentroid(*segment_cloud,centroid);
			  std::cout << "Centroid: " << centroid << std::endl;
			  
			  //Dertermine if the segment lies on the table plane solve ax * by *cz = d 
			  float value = coefficients->values[0] * centroid[0] + coefficients->values[1] * centroid[1] + coefficients->values[2] * centroid[2];
			  std::cout << "value: " << value << std::endl;
			  std::cout << "d: " << coefficients->values[3] << std::endl;
			  std::cout << "diff: " << coefficients->values[3] - std::abs(value)  << std::endl;
			  
			  pcl::PointCloud<pcl::PointXYZL>::Ptr temp_label_cloud (new pcl::PointCloud<pcl::PointXYZL> );
			
			  std::cout << "normal diff: " << std::abs(plane_theta -seg_theta) << std::endl;
			   
			  if((std::abs(coefficients->values[3] - std::abs(value)) > 0.001) &&
			     (std::abs(plane_theta -seg_theta) > 0.005)
			  ){
			  //  computePrincipalCurvature(segment_cloud)
			   CloudT::Ptr temp (new CloudT);
			            
			  //Deep copy
			  *temp = *segment_cloud; 
			  label_cloud(segment_cloud,temp_label_cloud, 1);
			  *labled_cloud += *temp_label_cloud;
			  PCL_WARN("Valid cluster found!\n");
			  // std::stringstream ss;
			  // std::cout << "Saving segment " << label <<  "..."<< std::endl;
			  // ss << "/home/thso/segment_" << label; ss << ".pcd";
			  // pcl::io::savePCDFile(ss.str(),*segment_cloud);
		    
			  clusters.push_back(*temp);
			 }else{
			    label_cloud(segment_cloud,temp_label_cloud, 2);
			   *labled_cloud += *temp_label_cloud;
			 }
			 
			  
		      }else{
			PCL_WARN("To less points in segment = %d\n", segment_cloud->points.size());
		      }
		  }else{
		      std::cerr << "No inliers!!!!" << std::endl;
		  }
		
	
	 segment_cloud->clear();
	
	  }
	PCL_WARN("Copy labled cloud\n");
	pcl::copyPointCloud(*labled_cloud, *lccp_labeled_cloud);
	 if(labled_cloud->points.size() > 0){
	   std::stringstream ss;
	   ss << "/home/thso/labled_cloud_ny.pcd";
	   pcl::io::savePCDFile(ss.str(),*labled_cloud);	  
	 }	  
	  
  //    }

 /*   
    //  if (pcl::getFieldIndex (*src_cloud, "label") >= 0)
   //       PCL_WARN ("Input cloud already has a label field. It will be overwritten by the lccp segmentation output.\n");
  
      ///Get all LCCP segments
      std::map<uint32_t, std::vector<uint32_t> > segment_supervoxel_map;   
      lccp.getSegmentSupervoxelMap(segment_supervoxel_map);
      uint32_t _num_of_segment = segment_supervoxel_map.size();
      std::cout << "Number of segment : " << _num_of_segment << std::endl;
      ///Initialize map to store inliers(segments)
      std::map<uint32_t, pcl::PointIndices::Ptr> segment_indice;
      for(uint32_t i = 0; i< _num_of_segment; i++){
	 pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	 segment_indice.insert(std::pair<uint32_t, pcl::PointIndices::Ptr>(i,inliers));
      }
      
      ///Iterate through LCCP labled cloud to copy label to inlier map
      pcl::PointCloud<pcl::PointXYZL>::iterator cloud_iter;
       std::map<uint32_t, uint32_t> pointcloud_label_to_new_label;
       int count = 0;
      for (cloud_iter = lccp_labeled_cloud->begin(); cloud_iter != lccp_labeled_cloud->end(); cloud_iter++) {
	//TODO: Kig på om label er ny. hvis: lig gamle lable i map på 1 til antal segmenters plads
	  uint32_t label =cloud_iter->label; 
	  if(pointcloud_label_to_new_label.count(label) == 0){ //The label dosen't exist
	      std::cout << "add label :" << count << std::endl;
             pointcloud_label_to_new_label.insert(std::pair<uint32_t,uint32_t>(label, count));
	     count++;
	  }else{ //label already exist. get the label position from pointcloud_label_to_new_label
	   uint32_t index = pointcloud_label_to_new_label.find(label)->second;
	 //  std::cout << "Save label :" << pointcloud_label_to_new_label.find(label)->first << " at index: " << index << std::endl;
	   if(index >= _num_of_segment ){
	     pcl::console::print_error("error: map index cannot exceed number of segments in cloud");
	     break;
	  }
	    //Map label to point
	    segment_indice.at(index)->indices.push_back(cloud_iter - lccp_labeled_cloud->begin());
	    
	  }
	
      }
      std::cout << "Number of labels : " << segment_indice.size() << std::endl;
      
      CloudT::Ptr segment_cloud (new CloudT);
      pcl::PointCloud<pcl::PointXYZL>::Ptr labled_cloud (new pcl::PointCloud<pcl::PointXYZL>);
      /// Extract full resolution point cloud for each segment 
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud (input_cloud_ptr);
      
      pcl::PointIndices::Ptr inliers_sac (new pcl::PointIndices);

      // Create the segmentation object
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.015);
      
      ///Segment iterator
   //   for(uint32_t i = 0; i<= segment_indice.size()-1; i++){
	  //Get all supervoxels in one segment
	  std::map<uint32_t, std::vector<uint32_t> >::iterator seg_iter;
	  std::cout << "segment_supervoxel_map size: " << segment_supervoxel_map.size() << std::endl;
	  for(seg_iter = segment_supervoxel_map.begin(); seg_iter != segment_supervoxel_map.end(); seg_iter++) {
		///First get the label
		uint32_t segment_label = seg_iter->first;
		std::cout << "-------------------VoxelID's for segment --------------------------" << std::endl;
		std::vector<uint32_t> supervoxels_in_segment = seg_iter->second;
	        std::cout << "Number of supervoxels in segment: " << supervoxels_in_segment.size() << std::endl;
		
		  ///Supervoxel iterator in one segment
		  float theta_avg = 0;
		  for(uint32_t j = 0; j< supervoxels_in_segment.size(); j++){
		    uint32_t id = supervoxels_in_segment.at(j);
		    pcl::PointNormal normal;
		    supervoxel_clusters.at(id)->getCentroidPointNormal(normal);
		    float radius, theta, phi = 0;
		    toSpherical(normal, radius, theta, phi); 
		    theta_avg += theta;
		    //std::cout << "Normal_x: " << normal.normal_x << " Normal_y: " << normal.normal_y << " Normal_z: " << normal.normal_z << std::endl;
		  //  std::cout << " theta: " << theta << " phi: " << phi << std::endl;
		  }
		  float samples = (float(supervoxels_in_segment.size())); 
		 // std::cout << " samples: " << samples << std::endl;
		 // std::cout << " theta_acc: " << theta_avg << std::endl;
		  std::cout << " theta_avg: " << theta_avg/samples << std::endl;
		  theta_avg = theta_avg/samples;
		  
		  int label = std::distance(segment_supervoxel_map.begin(), seg_iter);
		  //std::cout << " label inliers: " << label << std::endl;
		  pcl::PointIndices::Ptr inliers =segment_indice.at(label);
		  if (inliers->indices.size () != 0){
		      extract.setIndices (inliers);
		      extract.setNegative (false);
		      extract.filter (*segment_cloud);
		      if(segment_cloud->points.size() > 10){ //only fitting plane for clouds larger than 10 points
			 //Estimate the plane coefficients
			 pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			 seg.setInputCloud (segment_cloud);
			 seg.segment (*inliers_sac, *coefficients);
			 
			 if (inliers_sac->indices.size () == 0)
				PCL_ERROR ("Could not estimate a planar model for the given dataset.");
	      
			 std::cerr << "Model coefficients: "  << coefficients->values[0] << " "
								  << coefficients->values[1] << " "
								  << coefficients->values[2] << " "
								  << coefficients->values[3] << std::endl;
			 //coefficients.reset();
			 pcl::PointNormal plane_normal;
			 plane_normal.normal_x = coefficients->values[0];
			 plane_normal.normal_y = coefficients->values[1];
			 plane_normal.normal_z = coefficients->values[2];
			 float radius, theta,  phi = 0;
		         toSpherical(plane_normal, radius, theta, phi); 
			 std::cout <<"radius: " << radius << " theta: " << theta << " phi: " << phi << std::endl;
			   
			  //Compute centroid 
			  Eigen::Vector4d centroid;
			  pcl::compute3DCentroid(*segment_cloud,centroid);
			   // computePrincipalCurvature(segment_cloud);
			  std::cout << "Centroid: " << centroid << std::endl;
			    
			 pcl::PointCloud<pcl::PointXYZL>::Ptr temp_label_cloud (new pcl::PointCloud<pcl::PointXYZL> );
			 //Only add segments smaller than t= 5000
			 if((theta < 2.20f && centroid[2] < 1.20f) ){//&& (theta_avg < 2.20f)){//(theta < 1.50f || centroid[2] > 1.10) && (theta_avg > 2.00f)){
			      CloudT::Ptr temp (new CloudT);
			            
			      //Deep copy
			      *temp = *segment_cloud; 
			      label_cloud(segment_cloud,temp_label_cloud, 1);
			      *labled_cloud += *temp_label_cloud;
			      std::stringstream ss;
			      std::cout << "Saving segment " << label <<  "..."<< std::endl;
			 //     ss << "/home/thso/segment_" << label; ss << ".pcd";
			 //     pcl::io::savePCDFile(ss.str(),*segment_cloud);
			
			    //  clusters.push_back(*temp);
			 }else{
			    label_cloud(segment_cloud,temp_label_cloud, 2);
			   *labled_cloud += *temp_label_cloud;
			 }
		      }
		  }else{
		      std::cerr << "No inliers!!!!" << std::endl;
		  }
		
	
	 segment_cloud->clear();
	 pcl::io::savePCDFile("/home/thso/labled_cloud.pcd",*labled_cloud);	  
		  
		  
	  }
	*/  
  //    }
     
    if (output_specified)
    {
      PCL_INFO ("Saving output\n");
      if (add_label_field)
      {
        if (pcl::getFieldIndex (input_pointcloud2, "label") >= 0)
          PCL_WARN ("Input cloud already has a label field. It will be overwritten by the lccp segmentation output.\n");
        pcl::PCLPointCloud2 output_label_cloud2, output_concat_cloud2;
        pcl::toPCLPointCloud2 (*lccp_labeled_cloud, output_label_cloud2);
        pcl::concatenateFields (input_pointcloud2, output_label_cloud2, output_concat_cloud2);
        pcl::io::savePCDFile (outputname + "_out.pcd", output_concat_cloud2, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), save_binary_pcd);


      }
      else
        pcl::io::savePCDFile (outputname + "_out.pcd", *lccp_labeled_cloud, save_binary_pcd);

      if (sv_output_specified)
      {
        pcl::io::savePCDFile (outputname + "_svcloud.pcd", *sv_centroid_normal_cloud, save_binary_pcd);
      }
    }
  }
  else
  {
    PCL_ERROR ("ERROR:: Sizes of input cloud and labeled supervoxel cloud do not match. No output is produced.\n");
  }

  /// -----------------------------------|  Visualization  |-----------------------------------

  if (show_visualization)
  {
    /// Calculate visualization of adjacency graph
    // Using lines this would be VERY slow right now, because one actor is created for every line (may be fixed in future versions of PCL)
    // Currently this is a work-around creating a polygon mesh consisting of two triangles for each edge
    using namespace pcl;

    typedef LCCPSegmentation<PointT>::VertexIterator VertexIterator;
    typedef LCCPSegmentation<PointT>::AdjacencyIterator AdjacencyIterator;
    typedef LCCPSegmentation<PointT>::EdgeID EdgeID;

    std::set<EdgeID> edge_drawn;

    const unsigned char convex_color [3] = {255, 255, 255};
    const unsigned char concave_color [3] = {255, 0, 0};
    const unsigned char* color;
    
    //The vertices in the supervoxel adjacency list are the supervoxel centroids
    //This iterates through them, finding the edges
    std::pair<VertexIterator, VertexIterator> vertex_iterator_range;
    vertex_iterator_range = boost::vertices (sv_adjacency_list);

    /// Create a cloud of the voxelcenters and map: VertexID in adjacency graph -> Point index in cloud

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New (); 
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();     
    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    colors->SetNumberOfComponents (3);
    colors->SetName ("Colors");
    
    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();    
    for (VertexIterator itr = vertex_iterator_range.first; itr != vertex_iterator_range.second; ++itr)
    {
      const uint32_t sv_label = sv_adjacency_list[*itr];
      std::pair<AdjacencyIterator, AdjacencyIterator> neighbors = boost::adjacent_vertices (*itr, sv_adjacency_list);

      for (AdjacencyIterator itr_neighbor = neighbors.first; itr_neighbor != neighbors.second; ++itr_neighbor)
      {
        EdgeID connecting_edge = boost::edge (*itr, *itr_neighbor, sv_adjacency_list).first;  //Get the edge connecting these supervoxels
        if (sv_adjacency_list[connecting_edge].is_convex)
          color = convex_color;
        else
          color = concave_color;
        
        // two times since we add also two points per edge
        colors->InsertNextTupleValue (color);
        colors->InsertNextTupleValue (color);
        
        pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (sv_label);
        pcl::PointXYZRGBA vert_curr = supervoxel->centroid_;    
        
        
        const uint32_t sv_neighbor_label = sv_adjacency_list[*itr_neighbor];
        pcl::Supervoxel<PointT>::Ptr supervoxel_neigh = supervoxel_clusters.at (sv_neighbor_label);
        pcl::PointXYZRGBA vert_neigh = supervoxel_neigh->centroid_;
        
        points->InsertNextPoint (vert_curr.data);
        points->InsertNextPoint (vert_neigh.data);
          
        // Add the points to the dataset
        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();
        polyLine->GetPointIds ()->SetNumberOfIds (2);
        polyLine->GetPointIds ()->SetId (0, points->GetNumberOfPoints ()-2);
        polyLine->GetPointIds ()->SetId (1, points->GetNumberOfPoints ()-1);
        cells->InsertNextCell (polyLine);
      }
    }    
    polyData->SetPoints (points);
    // Add the lines to the dataset
    polyData->SetLines (cells);    
    
    polyData->GetPointData ()->SetScalars (colors);
        
    /// END: Calculate visualization of adjacency graph

    /// Configure Visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->registerKeyboardCallback (keyboardEventOccurred, 0);
    viewer->addCoordinateSystem(0.1,0,0,0,"",0);
    viewer->addPointCloud (lccp_labeled_cloud, "maincloud");

    /// Visualization Loop
    PCL_INFO ("Loading viewer\n");
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);

      /// Show Segmentation or Supervoxels
      viewer->updatePointCloud ( (show_supervoxels) ? sv_labeled_cloud : lccp_labeled_cloud, "maincloud");

      /// Show Normals
      if (normals_changed)
      {
        viewer->removePointCloud ("normals");
        if (show_normals)
        {
          viewer->addPointCloudNormals<pcl::PointNormal> (sv_centroid_normal_cloud, 1, normals_scale, "normals");
          normals_changed = false;
        }
      }
      /// Show Adjacency
      if (show_adjacency)
      {
        viewer->removeShape ("adjacency_graph");
        viewer->addModelFromPolyData (polyData, "adjacency_graph");
      }
      else
      {
        viewer->removeShape ("adjacency_graph");
      }

      if (show_help)
      {
        viewer->removeShape ("help_text");
        printText (viewer);
      }
      else
      {
        removeText (viewer);
        if (!viewer->updateText ("Press d to show help", 5, 10, 12, 1.0, 1.0, 1.0, "help_text"))
          viewer->addText ("Press d to show help", 5, 10, 12, 1.0, 1.0, 1.0, "help_text");
      }

      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
  }  /// END if (show_visualization)

  return (0);

}  /// END main

/// -------------------------| Definitions of helper functions|-------------------------

void
printText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg)
{
  std::string on_str = "ON";
  std::string off_str = "OFF";
  if (!viewer_arg->updateText ("Press (1-n) to show different elements (d) to disable this", 5, 72, 12, 1.0, 1.0, 1.0, "hud_text"))
        viewer_arg->addText ("Press (1-n) to show different elements", 5, 72, 12, 1.0, 1.0, 1.0, "hud_text");

  std::string temp = "(1) Supervoxel Normals, currently " + ( (show_normals) ? on_str : off_str);
  if (!viewer_arg->updateText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text"))
        viewer_arg->addText (temp, 5, 60, 10, 1.0, 1.0, 1.0, "normals_text");

  temp = "(2) Adjacency Graph, currently " + ( (show_adjacency) ? on_str : off_str) + "\n      White: convex; Red: concave";
  if (!viewer_arg->updateText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text"))
        viewer_arg->addText (temp, 5, 38, 10, 1.0, 1.0, 1.0, "graph_text");

  temp = "(3) Press to show " + ( (show_supervoxels) ? std::string ("SEGMENTATION") : std::string ("SUPERVOXELS"));
  if (!viewer_arg->updateText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text"))
        viewer_arg->addText (temp, 5, 26, 10, 1.0, 1.0, 1.0, "supervoxel_text");
  
  temp = "(4/5) Press to increase/decrease normals scale, currently " + boost::str (boost::format ("%.3f") % normals_scale);
  if (!viewer_arg->updateText (temp, 5, 14, 10, 1.0, 1.0, 1.0, "normals_scale_text"))
        viewer_arg->addText (temp, 5, 14, 10, 1.0, 1.0, 1.0, "normals_scale_text");
}

void
removeText (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_arg)
{
  viewer_arg->removeShape ("hud_text");
  viewer_arg->removeShape ("normals_text");
  viewer_arg->removeShape ("graph_text");
  viewer_arg->removeShape ("supervoxel_text");
  viewer_arg->removeShape ("normals_scale_text");
}
