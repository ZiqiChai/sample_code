/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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
 * $Id$
 */

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


#include <pcl/visualization/pcl_visualizer.h>

#include <numeric>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

std::string default_correspondence_type = "nn";
float default_output_resolution(300000);

float mean(std::vector<float> &v){
   // mean
   float sum2 = 0.0f;
   for(int i = 0; i<v.size();i++)
      sum2 += v.at(i);
     
   float sum = std::accumulate( v.begin(), v.end(), 0.0f )/ static_cast<float> (v.size());
   sum2 = sum2/static_cast<float> (v.size());
   std::cout << "accumulate sum: " << sum << std::endl;
   std::cout << "sqrt accumulate sum: " << sqrtf(sum) << std::endl;
   std::cout << "sum: " << sum2 << std::endl;
   return sum2;
}

float variance(std::vector<float> &v){
        double mean_ = mean(v);
        double temp = 0;
        for(std::vector<float>::iterator it = v.begin(); it !=v.end(); it++){
            temp += (mean_-*it)*(mean_-*it);
	}
        return temp/v.size();
    }

float stdDev(std::vector<float> &v){
     return std::sqrt(variance(v));
}


float median(std::vector<float> &v)
{
   std::size_t size = v.end() - v.begin();
   std::size_t middleIdx = size/2;
   
   std::vector<float>::iterator target = v.begin() + middleIdx;
   std::nth_element(v.begin(), target, v.end());
  
    size_t n = v.size() / 2;

    if(size % 2 != 0){ //Odd number of elements
      return  *target;
  }else{            //Even number of elements
     double a = *target;
    std::vector<float>::iterator targetNeighbor= target-1;
    std::nth_element(v.begin(), targetNeighbor, v.end());
    return (a+*targetNeighbor)/2.0;
  }
}


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

void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s source.pcd target.pcd output_accuracy_intensity.pcd output_completness_intensity.pcd <options>\n", argv[0]);
  print_info ("  where options are:\n");
  print_info ("\t-output_resolution W = How must weight shoud be applied to the final cloud. Larger resolution = smaller error visible \n");
  print_info ("	\t\t(default: ");
  print_value ("%s", default_output_resolution); print_info (")\n");
  print_info ("\t-correspondence X = the way of selecting the corresponding pair in the target cloud for the current point in the source cloud\n");
  print_info ("\t\toptions are: \n");
  print_info ("	\t\tindex = points with identical indices are paired together. Note: both clouds need to have the same number of points\n");
  print_info ("	\t\tnn = source point is paired with its nearest neighbor in the target cloud\n");
  print_info ("	\t\tnnplane = source point is paired with its projection on the plane determined by the nearest neighbor in the target cloud. Note: target cloud needs to contain normals\n");
  print_info ("	\t\t(default: ");
  print_value ("%s", default_correspondence_type.c_str ()); print_info (")\n");
}

bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
//  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud) < 0)
    return (false);
//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
//  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
computeCompletness (const pcl::PCLPointCloud2::ConstPtr &cloud_source, const pcl::PCLPointCloud2::ConstPtr &cloud_target,
         pcl::PCLPointCloud2 &output, std::string correspondence_type, float output_resolution)
{
  // Estimate
  TicToc tt;
  tt.tic ();
 PCL_INFO("Runnig with output_resolution = %f", output_resolution);
  PointCloud<PointXYZ>::Ptr xyz_source (new PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*cloud_source, *xyz_source);
  PointCloud<PointXYZ>::Ptr xyz_target (new PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*cloud_target, *xyz_target);

  PointCloud<PointXYZRGB>::Ptr output_xyzi (new PointCloud<PointXYZRGB> ());
  output_xyzi->points.resize (xyz_target->points.size ());
  output_xyzi->height = cloud_target->height;
  output_xyzi->width = cloud_target->width;

  
  float rmse = 0.0f;
  float median_error = 0.0f;
  float mean_error = 0.0f;
  float std_dev = 0.0f;
  int percent_count = 0;
  std::vector<float> distances;

  if (correspondence_type == "index")
  {
//    print_highlight (stderr, "Computing using the equal indices correspondence heuristic.\n");
/*
    if (xyz_source->points.size () != xyz_target->points.size ())
    {
      print_error ("Source and target clouds do not have the same number of points.\n");
      return;
    }

    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++point_i)
    {
      if (!pcl_isfinite (xyz_source->points[point_i].x) || !pcl_isfinite (xyz_source->points[point_i].y) || !pcl_isfinite (xyz_source->points[point_i].z))
        continue;
      if (!pcl_isfinite (xyz_target->points[point_i].x) || !pcl_isfinite (xyz_target->points[point_i].y) || !pcl_isfinite (xyz_target->points[point_i].z))
        continue;


      float dist = squaredEuclideanDistance (xyz_source->points[point_i], xyz_target->points[point_i]);
      rmse += dist;

      output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
      output_xyzi->points[point_i].intensity = dist * output_resolution;
    }
    rmse = sqrtf (rmse / static_cast<float> (xyz_source->points.size ()));
    */
  }
  else if (correspondence_type == "nn")
  {
    print_highlight (stderr, "Computing model Completness using the nearest neighbor correspondence heuristic.\n");

    KdTreeFLANN<PointXYZ>::Ptr tree (new KdTreeFLANN<PointXYZ> ());
    tree->setInputCloud (xyz_source);

    for (size_t point_i = 0; point_i < xyz_target->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (xyz_target->points[point_i].x) || !pcl_isfinite (xyz_target->points[point_i].y) || !pcl_isfinite (xyz_target->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (xyz_target->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      float euc_dist = euclideanDistance(xyz_target->points[point_i], xyz_source->points[point_nn_i]);
    //   std::cout << "euc_dist: " << euc_dist<< std::endl; 
      float dist = squaredEuclideanDistance (xyz_target->points[point_i], xyz_source->points[point_nn_i]);
      rmse += dist;
      if(euc_dist < 0.003){
	dist = 0.0f;
      }else{
	percent_count++;
      }
      distances.push_back(dist);
    }
    rmse = sqrtf (rmse / static_cast<float> (xyz_target->points.size ()));
    median_error = median(distances); 
    mean_error = mean(distances);
    std_dev = stdDev(distances);
     
     for (size_t point_i = 0; point_i < xyz_target->points.size (); ++ point_i)
    {
      
      output_xyzi->points[point_i].x = xyz_target->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_target->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_target->points[point_i].z;
        float res = (distances.at(point_i)/mean_error)*output_resolution * 128;
	//std::cout << "res: " << res << std::endl; 
      output_xyzi->points[point_i].r = res;
      output_xyzi->points[point_i].g = 255 -res;
      output_xyzi->points[point_i].b = 0;
     
    }

  }
  else if (correspondence_type == "nnplane")
  {
//    print_highlight (stderr, "Computing using the nearest neighbor plane projection correspondence heuristic.\n");
/*
    PointCloud<Normal>::Ptr normals_target (new PointCloud<Normal> ());
    fromPCLPointCloud2 (*cloud_target, *normals_target);

    KdTreeFLANN<PointXYZ>::Ptr tree (new KdTreeFLANN<PointXYZ> ());
    tree->setInputCloud (xyz_target);

    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (xyz_source->points[point_i].x) || !pcl_isfinite (xyz_source->points[point_i].y) || !pcl_isfinite (xyz_source->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (xyz_source->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      Eigen::Vector3f normal_target = normals_target->points[point_nn_i].getNormalVector3fMap (),
          point_source = xyz_source->points[point_i].getVector3fMap (),
          point_target = xyz_target->points[point_nn_i].getVector3fMap ();

      float dist = normal_target.dot (point_source - point_target);
      rmse += dist * dist;

      output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
      output_xyzi->points[point_i].intensity = (dist * dist) * output_resolution;
    }
    rmse = sqrtf (rmse / static_cast<float> (xyz_source->points.size ()));
    */
  }
  else
  {
    print_error ("Unrecognized correspondence type. Check legal arguments by using the -h option\n");
    return;
  }

  toPCLPointCloud2 (*output_xyzi, output);

  float coverage = 100-(((float)percent_count/(float)xyz_target->points.size())*100);
 // double coverage = 100.0f - ()*100.0f;

 // std::cout << "target_size: " << int(xyz_target->points.size()) << std::endl;
//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds]\n");
  print_highlight ("RMSE Error Completness: %e (%f mm)\n", rmse, rmse*1000);
  print_highlight ("Median Error Completness: %e (%f mm)\n", median_error, median_error*1000);
  print_highlight ("Mean Error Completness: %e (%f mm)\n", mean_error, mean_error*1000);
  print_highlight ("Std_dev Completness: %e (%f mm)\n", std_dev, std_dev*1000);
  print_highlight ("Coverage: %6.2f %%\n", coverage );
}

void
computeAccuracy (const pcl::PCLPointCloud2::ConstPtr &cloud_source, const pcl::PCLPointCloud2::ConstPtr &cloud_target,
         pcl::PCLPointCloud2 &output, std::string correspondence_type, float output_resolution)
{
  // Estimate
  TicToc tt;
  tt.tic ();

  PointCloud<PointXYZ>::Ptr xyz_source (new PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*cloud_source, *xyz_source);
  PointCloud<PointXYZ>::Ptr xyz_target (new PointCloud<PointXYZ> ());
  fromPCLPointCloud2 (*cloud_target, *xyz_target);

  PointCloud<PointXYZRGB>::Ptr output_xyzi (new PointCloud<PointXYZRGB> ());
  output_xyzi->points.resize (xyz_source->points.size ());
  output_xyzi->height = cloud_source->height;
  output_xyzi->width = cloud_source->width;

  float rmse = 0.0f;
  float median_error = 0.0f;
  float mean_error = 0.0f;
  float std_dev = 0.0f;
  std::vector<float> distances;
 
  if (correspondence_type == "index")
  {
//    print_highlight (stderr, "Computing using the equal indices correspondence heuristic.\n");

    if (xyz_source->points.size () != xyz_target->points.size ())
    {
      print_error ("Source and target clouds do not have the same number of points.\n");
      return;
    }

    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++point_i)
    {
      if (!pcl_isfinite (xyz_source->points[point_i].x) || !pcl_isfinite (xyz_source->points[point_i].y) || !pcl_isfinite (xyz_source->points[point_i].z))
        continue;
      if (!pcl_isfinite (xyz_target->points[point_i].x) || !pcl_isfinite (xyz_target->points[point_i].y) || !pcl_isfinite (xyz_target->points[point_i].z))
        continue;


      float dist = squaredEuclideanDistance (xyz_source->points[point_i], xyz_target->points[point_i]);
      rmse += dist;
      distances.push_back(dist);

      output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
 //     output_xyzi->points[point_i].intensity = dist * output_resolution;
    }
    rmse = sqrtf (rmse / static_cast<float> (xyz_source->points.size ()));
    median_error = median(distances); 
    mean_error = mean(distances);
    std_dev = stdDev(distances);
  }
  else if (correspondence_type == "nn")
  {
    print_highlight (stderr, "Computing using the nearest neighbor correspondence heuristic.\n");

    KdTreeFLANN<PointXYZ>::Ptr tree (new KdTreeFLANN<PointXYZ> ());
    tree->setInputCloud (xyz_target);

    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (xyz_source->points[point_i].x) || !pcl_isfinite (xyz_source->points[point_i].y) || !pcl_isfinite (xyz_source->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (xyz_source->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      float dist = squaredEuclideanDistance (xyz_source->points[point_i], xyz_target->points[point_nn_i]);
      //std::cout << "dist: " << dist << std::endl; 
  //    if(dist > 0.0001) dist = 0.0f;
      rmse += dist;
      distances.push_back(dist);
    }
    std::cout << "number of points: " << static_cast<float> (xyz_source->points.size ()) << std::endl;
    std::cout << "number of computed distances: " << static_cast<float> (distances.size()) << std::endl;
    std::cout << "rmse: " << rmse << std::endl;
    rmse = sqrtf (rmse / static_cast<float> (xyz_source->points.size ()));
    median_error = median(distances);  
    mean_error = mean(distances);
    std_dev = stdDev(distances);
    
    
    
   
     
    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i)
    {
   //    std::cout << "dist: " << distances.at(point_i) << std::endl; 
      output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
        float res = (distances.at(point_i)/mean_error)*output_resolution * 128;
      output_xyzi->points[point_i].r = res;
      output_xyzi->points[point_i].g = 255 -res;
      output_xyzi->points[point_i].b = 0;
     
    }
  }
  else if (correspondence_type == "nnplane")
  {
//    print_highlight (stderr, "Computing using the nearest neighbor plane projection correspondence heuristic.\n");

    PointCloud<Normal>::Ptr normals_target (new PointCloud<Normal> ());
    fromPCLPointCloud2 (*cloud_target, *normals_target);

    KdTreeFLANN<PointXYZ>::Ptr tree (new KdTreeFLANN<PointXYZ> ());
    tree->setInputCloud (xyz_target);

    for (size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i)
    {
      if (!pcl_isfinite (xyz_source->points[point_i].x) || !pcl_isfinite (xyz_source->points[point_i].y) || !pcl_isfinite (xyz_source->points[point_i].z))
        continue;

      std::vector<int> nn_indices (1);
      std::vector<float> nn_distances (1);
      if (!tree->nearestKSearch (xyz_source->points[point_i], 1, nn_indices, nn_distances))
        continue;
      size_t point_nn_i = nn_indices.front();

      Eigen::Vector3f normal_target = normals_target->points[point_nn_i].getNormalVector3fMap (),
          point_source = xyz_source->points[point_i].getVector3fMap (),
          point_target = xyz_target->points[point_nn_i].getVector3fMap ();

      float dist = normal_target.dot (point_source - point_target);
      rmse += dist * dist;
      distances.push_back(dist);

      output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
      output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
      output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
   //   output_xyzi->points[point_i].intensity = (dist * dist) * output_resolution;
    }
    rmse = sqrtf (rmse / static_cast<float> (xyz_source->points.size ()));
    median_error = median(distances); 
    mean_error = mean(distances);
    std_dev = stdDev(distances);
  }
  else
  {
    print_error ("Unrecognized correspondence type. Check legal arguments by using the -h option\n");
    return;
  }

  toPCLPointCloud2 (*output_xyzi, output);

//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds]\n");
  print_highlight ("RMSE Error: %e (%f mm)\n", rmse, rmse*1000);
  print_highlight ("Median Error: %e (%f mm)\n", median_error, median_error*1000);
  print_highlight ("Mean Error: %e (%f mm)\n", mean_error, mean_error*1000);
  print_highlight ("Std_dev: %e (%f mm)\n", std_dev, std_dev*1000);
}

void
saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &output)
{
  TicToc tt;
  tt.tic ();

//  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());

  pcl::io::savePCDFile (filename, output);

//  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" seconds : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}


/* ---[ */
int
main (int argc, char** argv)
{
//  print_info ("Compute the differences between two point clouds and visualizing them as an output intensity cloud. For more information, use: %s -h\n", argv[0]);

  if (argc < 4)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd files
  std::vector<int> p_file_indices;
  p_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  if (p_file_indices.size () != 4)
  {
    print_error ("Need two input PCD files and two output PCD file to continue.\n");
    return (-1);
  }

  // Command line parsing
  std::string correspondence_type = default_correspondence_type;
  parse_argument (argc, argv, "-correspondence", correspondence_type);
  
  float output_resolution = default_output_resolution;
  parse_argument (argc, argv, "-output_resolution", output_resolution);

  // Load the first file
  pcl::PCLPointCloud2::Ptr cloud_source (new pcl::PCLPointCloud2 ());
  if (!loadCloud (argv[p_file_indices[0]], *cloud_source))
    return (-1);
   print_highlight("Source cloud loaded with an average point resolution = %f mm\n", ComputeCloudResolution(cloud_source)*1000);
  // Load the second file
  pcl::PCLPointCloud2::Ptr cloud_target (new pcl::PCLPointCloud2 ());
  if (!loadCloud (argv[p_file_indices[1]], *cloud_target))
    return (-1);
  print_highlight("Target cloud loaded with an average point resolution = %f mm\n", ComputeCloudResolution(cloud_target)*1000);
 
  
  pcl::PCLPointCloud2 output;
  // Perform the feature estimation
  computeAccuracy (cloud_source, cloud_target, output, correspondence_type, output_resolution);
  saveCloud (argv[p_file_indices[2]], output);
 
  computeCompletness (cloud_source, cloud_target, output, correspondence_type, output_resolution);

  // Output the third file
  saveCloud (argv[p_file_indices[3]], output);
}
