/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
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
 *
 */

#ifndef PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_CYLINDER_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_CYLINDER_H_

#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/common/common.h> // for getAngle3D
#include <pcl/common/concatenate.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelCylinder<PointT, PointNT>::isSampleGood (const Indices &samples) const
{
  if (samples.size () != sample_size_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::isSampleGood] Wrong number of samples (is %lu, should be %lu)!\n", samples.size (), sample_size_);
    return (false);
  }

  // Make sure that the two sample points are not identical
  if (
      std::abs ((*input_)[samples[0]].x - (*input_)[samples[1]].x) <= std::numeric_limits<float>::epsilon ()
    &&
      std::abs ((*input_)[samples[0]].y - (*input_)[samples[1]].y) <= std::numeric_limits<float>::epsilon ()
    &&
      std::abs ((*input_)[samples[0]].z - (*input_)[samples[1]].z) <= std::numeric_limits<float>::epsilon ())
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::isSampleGood] The two sample points are (almost) identical!\n");
    return (false);
  }

  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelCylinder<PointT, PointNT>::computeModelCoefficients (
      const Indices &samples, Eigen::VectorXf &model_coefficients) const
{
  // Make sure that the samples are valid
  if (!isSampleGood (samples))
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::computeModelCoefficients] Invalid set of samples given!\n");
    return (false);
  }

  if (!normals_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::computeModelCoefficients] No input dataset containing normals was given! Use setInputNormals\n");
    return (false);
  }

  // Use 3D maps for points and normals
  const auto p1 = (*input_)[samples[0]].getVector3fMap();
  const auto p2 = (*input_)[samples[1]].getVector3fMap();
  const auto n1 = (*normals_)[samples[0]].getNormalVector3fMap();
  const auto n2 = (*normals_)[samples[1]].getNormalVector3fMap();

  // Compute closest points on the two (almost) parallel lines defined by (p1 + n1 + s*n1) and (p2 + t*n2)
  const Eigen::Vector3f w = n1 + p1 - p2;
  const float b = n1.dot (n2);
  const float d = n1.dot (w);
  const float e = n2.dot (w);
  const float denominator = 1.0f - b * b; // assumes unit normals; consistent with original implementation
  float sc, tc;
  if (denominator < 1e-8f) // The lines are almost parallel
  {
    sc = 0.0f;
    tc = (b > 1.0f ? d / b : e); // Use the largest denominator (matches original logic)
  }
  else
  {
    sc = (b * e - d) / denominator;
    tc = (e - b * d) / denominator;
  }

  // point_on_axis and axis_direction
  const Eigen::Vector3f line_pt3 = p1 + n1 + sc * n1;
  const Eigen::Vector3f line_dir3 = (p2 + tc * n2 - line_pt3).normalized();

  model_coefficients.resize (model_size_);
  model_coefficients[0] = line_pt3[0];
  model_coefficients[1] = line_pt3[1];
  model_coefficients[2] = line_pt3[2];
  model_coefficients[3] = line_dir3[0];
  model_coefficients[4] = line_dir3[1];
  model_coefficients[5] = line_dir3[2];

  // cylinder radius: distance from p1 to the axis
  const Eigen::Vector3f v = p1 - line_pt3;
  const float vdotd = v.dot(line_dir3);
  const float radius = (v - vdotd * line_dir3).norm();
  model_coefficients[6] = radius;

  if (model_coefficients[6] > radius_max_ || model_coefficients[6] < radius_min_)
    return (false);

  PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::computeModelCoefficients] Model is (%g,%g,%g,%g,%g,%g,%g).\n",
             model_coefficients[0], model_coefficients[1], model_coefficients[2], model_coefficients[3],
             model_coefficients[4], model_coefficients[5], model_coefficients[6]);
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelCylinder<PointT, PointNT>::getDistancesToModel (
      const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) const
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    distances.clear ();
    return;
  }

  distances.resize (indices_->size ());

  // Work with 3D parts to avoid any interaction with the homogeneous component
  const Eigen::Vector3f line_pt (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  const Eigen::Vector3f line_dir_raw (model_coefficients[3], model_coefficients[4], model_coefficients[5]);
  Eigen::Vector3f line_dir = line_dir_raw;
  const float dir_norm = line_dir.norm();
  if (dir_norm > 0.0f)
    line_dir /= dir_norm;

  const double radius = static_cast<double>(model_coefficients[6]);
  const double ndw = static_cast<double>(normal_distance_weight_);
  const double one_minus_weight = 1.0 - ndw;

  // Iterate through the 3d points and calculate the distances from them to the cylinder
  for (std::size_t i = 0; i < indices_->size (); ++i)
  {
    // Vector from a point on the axis to the query point
    const Eigen::Vector3f pt = (*input_)[(*indices_)[i]].getVector3fMap();
    const Eigen::Vector3f v = pt - line_pt;

    // Decompose v into parallel and perpendicular components wrt the axis (d is unit)
    const float vdotd = v.dot(line_dir);
    const Eigen::Vector3f v_perp = v - vdotd * line_dir; // perpendicular from axis to point

    // Euclidean term: distance from axis minus cylinder radius
    const double radial = static_cast<double>(v_perp.norm());
    const double weighted_euclid_dist = one_minus_weight * std::abs(radial - radius);

    // Angular term: angle between point normal and direction from axis to point
    double d_normal_term = 0.0;
    if (ndw > 0.0)
    {
      // Guard against zero perpendicular (undefined direction)
      const float vperp_sq = v_perp.squaredNorm();
      if (vperp_sq > std::numeric_limits<float>::epsilon())
      {
        const Eigen::Vector3f n3 = (*normals_)[(*indices_)[i]].getNormalVector3fMap();
        double d_normal = std::abs (getAngle3D (n3, v_perp));
        d_normal = (std::min) (d_normal, M_PI - d_normal);
        d_normal_term = ndw * d_normal;
      }
      // else: leave d_normal_term = 0
    }

    distances[i] = std::abs (d_normal_term + weighted_euclid_dist);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelCylinder<PointT, PointNT>::selectWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold, Indices &inliers)
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    inliers.clear ();
    return;
  }

  inliers.clear ();
  error_sqr_dists_.clear ();
  inliers.reserve (indices_->size ());
  error_sqr_dists_.reserve (indices_->size ());

  // Extract axis point and direction (3D)
  const Eigen::Vector3f line_pt (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  const Eigen::Vector3f line_dir_raw(model_coefficients[3], model_coefficients[4], model_coefficients[5]);
  // Normalize direction defensively for correct projection math
  Eigen::Vector3f line_dir = line_dir_raw;
  const float dir_norm = line_dir.norm();
  if (dir_norm > 0.0f)
    line_dir /= dir_norm;

  const double radius = static_cast<double>(model_coefficients[6]);
  const double ndw = static_cast<double>(normal_distance_weight_);
  const double one_minus_weight = 1.0 - ndw;
  const bool use_euclid = one_minus_weight > 0.0;
  const double euclid_thresh = use_euclid ? (threshold / one_minus_weight) : 0.0;
  const double rmin_sq = use_euclid ? std::max(0.0, (radius - euclid_thresh)) * std::max(0.0, (radius - euclid_thresh)) : 0.0;
  const double rmax_sq = use_euclid ? (radius + euclid_thresh) * (radius + euclid_thresh) : 0.0;

  // Iterate through the 3d points and calculate the distances from them to the cylinder
  for (std::size_t i = 0; i < indices_->size (); ++i)
  {
    const auto idx = (*indices_)[i];

    // Vector from a point on the axis to the query point
    const Eigen::Vector3f pt = (*input_)[idx].getVector3fMap();
    const Eigen::Vector3f v = pt - line_pt;

    // Compute perpendicular from axis to point: v_perp = v - (v·d) d (d is unit)
    const float vdotd = v.dot(line_dir);
    const Eigen::Vector3f v_perp = v - vdotd * line_dir;

    // Early-out using squared bounds to avoid sqrt when possible
    double weighted_euclid_dist = 0.0;
    if (use_euclid)
    {
      const double radial_sq = static_cast<double>(v_perp.squaredNorm());
      if (radial_sq < rmin_sq || radial_sq > rmax_sq)
        continue;
      // Survived: compute exact Euclidean term only now
      const double radial = std::sqrt(radial_sq);
      weighted_euclid_dist = one_minus_weight * std::abs(radial - radius);
      if (weighted_euclid_dist > threshold) // Defensive check
        continue;
    }

    // Angular term (only if needed)
    double d_normal_term = 0.0;
    if (ndw > 0.0)
    {
      const float vperp_sq = v_perp.squaredNorm();
      if (vperp_sq > std::numeric_limits<float>::epsilon())
      {
        // Use 3D normals and vectors for angle computation
        const Eigen::Vector3f n3 = (*normals_)[idx].getNormalVector3fMap();
        double d_normal = std::abs(getAngle3D(n3, v_perp));
        d_normal = (std::min) (d_normal, M_PI - d_normal);
        d_normal_term = ndw * d_normal;
      }
      // else: leave d_normal_term = 0
    }

    const double distance = std::abs (d_normal_term + weighted_euclid_dist);
    if (distance < threshold)
    {
      // Returns the indices of the points whose distances are smaller than the threshold
      inliers.push_back (idx);
      error_sqr_dists_.push_back (distance);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> std::size_t
pcl::SampleConsensusModelCylinder<PointT, PointNT>::countWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold) const
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
    return (0);

  std::size_t nr_p = 0;

  // Use 3D vectors for performance
  const Eigen::Vector3f line_pt (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  const Eigen::Vector3f line_dir_raw(model_coefficients[3], model_coefficients[4], model_coefficients[5]);
  // Normalize direction defensively for correct projection math
  Eigen::Vector3f line_dir = line_dir_raw;
  const float dir_norm = line_dir.norm();
  if (dir_norm > 0.0f)
    line_dir /= dir_norm;

  const double radius = static_cast<double>(model_coefficients[6]);
  const double ndw = static_cast<double>(normal_distance_weight_);
  const double one_minus_weight = 1.0 - ndw;
  const bool use_euclid = one_minus_weight > 0.0;
  const double euclid_thresh = use_euclid ? (threshold / one_minus_weight) : 0.0;
  const double rmin_sq = use_euclid ? std::max(0.0, (radius - euclid_thresh)) * std::max(0.0, (radius - euclid_thresh)) : 0.0;
  const double rmax_sq = use_euclid ? (radius + euclid_thresh) * (radius + euclid_thresh) : 0.0;

  // Iterate through the 3d points and calculate the distances from them to the cylinder
  for (std::size_t i = 0; i < indices_->size (); ++i)
  {
    const auto idx = (*indices_)[i];

    // Vector from a point on the axis to the query point
    const Eigen::Vector3f pt = (*input_)[idx].getVector3fMap();
    const Eigen::Vector3f v = pt - line_pt;

    // Perpendicular component to axis (d is unit)
    const float vdotd = v.dot(line_dir);
    const Eigen::Vector3f v_perp = v - vdotd * line_dir;

    // Early-out using squared bounds to avoid sqrt when possible
    double weighted_euclid_dist = 0.0;
    if (use_euclid)
    {
      const double radial_sq = static_cast<double>(v_perp.squaredNorm());
      if (radial_sq < rmin_sq || radial_sq > rmax_sq)
        continue;
      const double radial = std::sqrt(radial_sq);
      weighted_euclid_dist = one_minus_weight * std::abs(radial - radius);
      if (weighted_euclid_dist > threshold)
        continue;
    }

    // Angular term (if enabled)
    double d_normal_term = 0.0;
    if (ndw > 0.0)
    {
      const float vperp_sq = v_perp.squaredNorm();
      if (vperp_sq > std::numeric_limits<float>::epsilon())
      {
        const Eigen::Vector3f n3 = (*normals_)[idx].getNormalVector3fMap();
        double d_normal = std::abs(getAngle3D(n3, v_perp));
        d_normal = (std::min) (d_normal, M_PI - d_normal);
        d_normal_term = ndw * d_normal;
      }
      // else: leave d_normal_term = 0
    }

    if (std::abs (d_normal_term + weighted_euclid_dist) < threshold)
      nr_p++;
  }
  return (nr_p);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelCylinder<PointT, PointNT>::optimizeModelCoefficients (
      const Indices &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients) const
{
  optimized_coefficients = model_coefficients;

  // Needs a set of valid model coefficients
  if (!isModelValid (model_coefficients))
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::optimizeModelCoefficients] Given model is invalid!\n");
    return;
  }

  // Need more than the minimum sample size to make a difference
  if (inliers.size () <= sample_size_)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder:optimizeModelCoefficients] Not enough inliers found to optimize model coefficients (%lu)! Returning the same coefficients.\n", inliers.size ());
    return;
  }

  Eigen::ArrayXf pts_x(inliers.size());
  Eigen::ArrayXf pts_y(inliers.size());
  Eigen::ArrayXf pts_z(inliers.size());
  std::size_t pos = 0;
  for(const auto& index : inliers) {
    pts_x[pos] = (*input_)[index].x;
    pts_y[pos] = (*input_)[index].y;
    pts_z[pos] = (*input_)[index].z;
    ++pos;
  }
  pcl::internal::optimizeModelCoefficientsCylinder(optimized_coefficients, pts_x, pts_y, pts_z);
  
  PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::optimizeModelCoefficients] Initial solution: %g %g %g %g %g %g %g \nFinal solution: %g %g %g %g %g %g %g\n",
             model_coefficients[0], model_coefficients[1], model_coefficients[2], model_coefficients[3],
             model_coefficients[4], model_coefficients[5], model_coefficients[6], optimized_coefficients[0], optimized_coefficients[1], optimized_coefficients[2], optimized_coefficients[3], optimized_coefficients[4], optimized_coefficients[5], optimized_coefficients[6]);
    
  Eigen::Vector3f line_dir (optimized_coefficients[3], optimized_coefficients[4], optimized_coefficients[5]);
  line_dir.normalize ();
  optimized_coefficients[3] = line_dir[0];
  optimized_coefficients[4] = line_dir[1];
  optimized_coefficients[5] = line_dir[2];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelCylinder<PointT, PointNT>::projectPointToCylinder (
      const Eigen::Vector4f &pt, const Eigen::VectorXf &model_coefficients, Eigen::Vector4f &pt_proj) const
{
  // Use 3D math for projection
  const Eigen::Vector3f line_pt (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  const Eigen::Vector3f line_dir (model_coefficients[3], model_coefficients[4], model_coefficients[5]);

  // Project point onto axis line (not necessarily unit direction)
  const Eigen::Vector3f p3 (pt[0], pt[1], pt[2]);
  const Eigen::Vector3f diff = p3 - line_pt;
  const float dirdotdir = line_dir.dot(line_dir);
  const float k = dirdotdir > 0.0f ? diff.dot(line_dir) / dirdotdir : 0.0f;
  Eigen::Vector3f proj = line_pt + k * line_dir;

  // Direction from axis to point on cylinder surface
  Eigen::Vector3f dir = p3 - proj;
  const float dir_norm = dir.norm();
  if (dir_norm > std::numeric_limits<float>::epsilon())
    dir /= dir_norm;

  proj += dir * model_coefficients[6];

  pt_proj[0] = proj[0];
  pt_proj[1] = proj[1];
  pt_proj[2] = proj[2];
  pt_proj[3] = 0.0f;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> void
pcl::SampleConsensusModelCylinder<PointT, PointNT>::projectPoints (
      const Indices &inliers, const Eigen::VectorXf &model_coefficients, PointCloud &projected_points, bool copy_data_fields) const
{
  // Needs a valid set of model coefficients
  if (!isModelValid (model_coefficients))
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::projectPoints] Given model is invalid!\n");
    return;
  }

  projected_points.header = input_->header;
  projected_points.is_dense = input_->is_dense;

  // Work in 3D
  const Eigen::Vector3f line_pt (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  const Eigen::Vector3f line_dir (model_coefficients[3], model_coefficients[4], model_coefficients[5]);
  const float inv_dirdotdir = 1.0f / line_dir.dot (line_dir);

  // Copy all the data fields from the input cloud to the projected one?
  if (copy_data_fields)
  {
    // Allocate enough space and copy the basics
    projected_points.resize (input_->size ());
    projected_points.width    = input_->width;
    projected_points.height   = input_->height;

    using FieldList = typename pcl::traits::fieldList<PointT>::type;
    // Iterate over each point
    for (std::size_t i = 0; i < projected_points.size (); ++i)
      // Iterate over each dimension
      pcl::for_each_type <FieldList> (NdConcatenateFunctor <PointT, PointT> ((*input_)[i], projected_points[i]));

    // Iterate through the 3d points and calculate the distances from them to the cylinder
    for (const auto &inlier : inliers)
    {
      const Eigen::Vector3f p ((*input_)[inlier].x,
                               (*input_)[inlier].y,
                               (*input_)[inlier].z);

      const float k = ((p - line_pt).dot (line_dir)) * inv_dirdotdir;

      Eigen::Vector3f pp3 = line_pt + k * line_dir;

      Eigen::Vector3f dir = p - pp3;
      const float dir_norm = dir.norm();
      if (dir_norm > std::numeric_limits<float>::epsilon())
        dir /= dir_norm;

      // Calculate the projection of the point onto the cylinder
      pp3 += dir * model_coefficients[6];
      projected_points[inlier].x = pp3[0];
      projected_points[inlier].y = pp3[1];
      projected_points[inlier].z = pp3[2];
    }
  }
  else
  {
    // Allocate enough space and copy the basics
    projected_points.resize (inliers.size ());
    projected_points.width    = inliers.size ();
    projected_points.height   = 1;

    using FieldList = typename pcl::traits::fieldList<PointT>::type;
    // Iterate over each point
    for (std::size_t i = 0; i < inliers.size (); ++i)
      // Iterate over each dimension
      pcl::for_each_type <FieldList> (NdConcatenateFunctor <PointT, PointT> ((*input_)[inliers[i]], projected_points[i]));

    // Iterate through the 3d points and calculate the distances from them to the cylinder
    for (std::size_t i = 0; i < inliers.size (); ++i)
    {
      const Eigen::Vector3f p ((*input_)[inliers[i]].x,
                               (*input_)[inliers[i]].y,
                               (*input_)[inliers[i]].z);

      const float k = ((p - line_pt).dot (line_dir)) * inv_dirdotdir;
      Eigen::Vector3f pp3 = line_pt + k * line_dir;

      Eigen::Vector3f dir = p - pp3;
      const float dir_norm = dir.norm();
      if (dir_norm > std::numeric_limits<float>::epsilon())
        dir /= dir_norm;

      // Calculate the projection of the point onto the cylinder
      pp3 += dir * model_coefficients[6];
      projected_points[i].x = pp3[0];
      projected_points[i].y = pp3[1];
      projected_points[i].z = pp3[2];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool
pcl::SampleConsensusModelCylinder<PointT, PointNT>::doSamplesVerifyModel (
      const std::set<index_t> &indices, const Eigen::VectorXf &model_coefficients, const double threshold) const
{
  // Needs a valid model coefficients
  if (!isModelValid (model_coefficients))
  {
    PCL_ERROR ("[pcl::SampleConsensusModelCylinder::doSamplesVerifyModel] Given model is invalid!\n");
    return (false);
  }

  for (const auto &index : indices)
  {
    // Approximate the distance from the point to the cylinder as the difference between
    // dist(point,cylinder_axis) and cylinder radius
    // @note need to revise this.
    Eigen::Vector4f pt ((*input_)[index].x, (*input_)[index].y, (*input_)[index].z, 0.0f);
    if (std::abs (pointToLineDistance (pt, model_coefficients) - model_coefficients[6]) > threshold)
      return (false);
  }

  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> double
pcl::SampleConsensusModelCylinder<PointT, PointNT>::pointToLineDistance (
      const Eigen::Vector4f &pt, const Eigen::VectorXf &model_coefficients) const
{
  Eigen::Vector4f line_pt  (model_coefficients[0], model_coefficients[1], model_coefficients[2], 0.0f);
  Eigen::Vector4f line_dir (model_coefficients[3], model_coefficients[4], model_coefficients[5], 0.0f);
  return sqrt(pcl::sqrPointToLineDistance (pt, line_pt, line_dir));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename PointNT> bool 
pcl::SampleConsensusModelCylinder<PointT, PointNT>::isModelValid (const Eigen::VectorXf &model_coefficients) const
{
  if (!SampleConsensusModel<PointT>::isModelValid (model_coefficients))
    return (false);

  // Check against template, if given
  if (eps_angle_ > 0.0)
  {
    // Obtain the cylinder direction
    const Eigen::Vector3f coeff(model_coefficients[3], model_coefficients[4], model_coefficients[5]);

    double angle_diff = std::abs (getAngle3D (axis_, coeff));
    angle_diff = (std::min) (angle_diff, M_PI - angle_diff);
    // Check whether the current cylinder model satisfies our angle threshold criterion with respect to the given axis
    if (angle_diff > eps_angle_)
    {
      PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::isModelValid] Angle between cylinder direction and given axis is too large.\n");
      return (false);
    }
  }

  if (radius_min_ != -std::numeric_limits<double>::max() && model_coefficients[6] < radius_min_)
  {
    PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::isModelValid] Radius is too small: should be larger than %g, but is %g.\n",
               radius_min_, model_coefficients[6]);
    return (false);
  }
  if (radius_max_ != std::numeric_limits<double>::max() && model_coefficients[6] > radius_max_)
  {
    PCL_DEBUG ("[pcl::SampleConsensusModelCylinder::isModelValid] Radius is too big: should be smaller than %g, but is %g.\n",
               radius_max_, model_coefficients[6]);
    return (false);
  }

  return (true);
}

#define PCL_INSTANTIATE_SampleConsensusModelCylinder(PointT, PointNT)	template class PCL_EXPORTS pcl::SampleConsensusModelCylinder<PointT, PointNT>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_CYLINDER_H_

