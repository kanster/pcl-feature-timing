/**
 * time consumption comparsion analysis for 3D keypoints in Point Cloud Library
 * @methods
 *    - normal 3d
 *    - normal 3d omp
 *    -
 * @author Kanzhi Wu
 */


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/harris_6d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/smoothed_surfaces_keypoint.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <string>
#include <vector>


void print_usage( char * argv ) {
  pcl::console::print_error( "Syntax is: %s rgb_image depth_image\n", argv );
  std::cout << "\n\n";
}


/// convert images to xyzrgb point cloud
///
/// rgb_img     -- 3 channels, 8bit rgb image
/// depth_img   -- 16 bit depth image
/// param       -- cx, cy, fx, fy
///
pcl::PointCloud<pcl::PointXYZRGB>::Ptr img2cloud(cv::Mat rgb_img, cv::Mat depth_img, float * param ) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGB>() );
  cloud->width = rgb_img.cols;
  cloud->height = rgb_img.rows;
  cloud->is_dense = false;
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  for ( int y = 0; y < rgb_img.rows; ++ y ) {
    for ( int x = 0; x < rgb_img.cols; ++ x ) {
      pcl::PointXYZRGB pt;
      pt.b = rgb_img.at<cv::Vec3b>(y,x)[0];
      pt.g = rgb_img.at<cv::Vec3b>(y,x)[1];
      pt.r = rgb_img.at<cv::Vec3b>(y,x)[2];
      if (depth_img.at<unsigned short>(y, x) == 0) {
        pt.x = bad_point; //std::numeric_limits<float>::quiet_NaN();
        pt.y = bad_point; //std::numeric_limits<float>::quiet_NaN();
        pt.z = bad_point; //std::numeric_limits<float>::quiet_NaN();
      }
      else {
        pt.z = depth_img.at<unsigned short>(y, x)/1000.;
        pt.x = pt.z*(x-param[0])/param[2];
        pt.y = pt.z*(y-param[1])/param[3];
      }
      cloud->points.push_back(pt);
    }
  }
  return cloud;
}



/// Harris3d keypoints detector
///
/// cloud       -- input point cloud
///
pcl::PointCloud<pcl::PointXYZI>::Ptr harris3d( pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                               pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::ResponseMethod method ) {
  pcl::console::TicToc tt;
  tt.tic();

  // detector
  pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::Ptr harris_detector(
        new pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>( method ) );

  harris_detector->setNonMaxSupression( true );
  harris_detector->setRadius( 0.03f );
  harris_detector->setRadiusSearch( 0.03f );


  pcl::PointCloud<pcl::PointXYZI>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZI>() );

  harris_detector->setInputCloud( cloud );
  harris_detector->compute( *kpts );
  double t = tt.toc();
  std::string method_name;
  switch( method ) {
  case pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZ>::HARRIS:
    method_name = "HARRIS";
    break;
  case pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZ>::TOMASI:
    method_name = "TOMASI";
    break;
  case pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZ>::NOBLE:
    method_name = "NOBLE";
    break;
  case pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZ>::LOWE:
    method_name = "LOWE";
    break;
  case pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZ>::CURVATURE:
    method_name = "CURVATURE";
    break;
  }
  pcl::console::print_value( "Harris3D (%s) takes %.3f for extractiing %d keypoints\n", method_name.c_str(), t, (int)kpts->size() );

  return kpts;
}



/// Harris6d keypoints detector
///
/// cloud       -- input point cloud
///
pcl::PointCloud<pcl::PointXYZI>::Ptr harris6d( pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud ) {
  pcl::console::TicToc tt;
  tt.tic();

  // detector
  pcl::HarrisKeypoint6D<pcl::PointXYZRGB, pcl::PointXYZI>::Ptr harris_detector(
        new pcl::HarrisKeypoint6D<pcl::PointXYZRGB, pcl::PointXYZI>(  ) );

  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree( new pcl::search::KdTree<pcl::PointXYZRGB>() );

  harris_detector->setSearchMethod( tree );
  harris_detector->setNonMaxSupression( true );
  harris_detector->setRadius( 0.03f );
  harris_detector->setRadiusSearch( 0.03f );


  pcl::PointCloud<pcl::PointXYZI>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZI>() );

  harris_detector->setInputCloud( cloud );
  harris_detector->compute( *kpts );
  double t = tt.toc();
  pcl::console::print_value( "Harris6D takes %.3f for extractiing %d keypoints\n", t, (int)kpts->size() );

  return kpts;
}

/// compute point cloud resolution
///
/// cloud       -- input point cloud
///
double compute_cloud_resolution (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZRGB> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i) {
    if (! pcl_isfinite ((*cloud)[i].x)) {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2) {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0) {
    res /= n_points;
  }
  return res;
}


/// iss3d keypoints detector
///
/// cloud       -- input point cloud
///
pcl::PointCloud<pcl::PointXYZRGB>::Ptr iss3d( pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud ) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree( new pcl::search::KdTree<pcl::PointXYZRGB>() );

  double cloud_resolution = compute_cloud_resolution( cloud );

  iss_detector.setSearchMethod (tree);
  iss_detector.setSalientRadius (6 * cloud_resolution);
  iss_detector.setNonMaxRadius (4 * cloud_resolution);

  iss_detector.setThreshold21 (0.975);
  iss_detector.setThreshold32 (0.975);
  iss_detector.setMinNeighbors (5);
  iss_detector.setNumberOfThreads (1);
  iss_detector.setInputCloud (cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZRGB>() );
  iss_detector.compute(*kpts);

  double t = tt.toc();
  pcl::console::print_value( "ISS3D takes %.3f for extractiing %d keypoints\n", t, (int)kpts->size() );

  return kpts;
}


int main( int argc, char ** argv ) {
  pcl::console::setVerbosityLevel( pcl::console::L_ERROR );

  if ( argc < 2 ) {
    print_usage( argv[0] );
    return (-1);
  }

  if (pcl::console::find_switch( argc, argv, "-help" ) ||
      pcl::console::find_switch( argc, argv, "-h" )) {
    print_usage( argv[0] );
  }

  // load images
  std::string rgbp = argv[1], depthp = argv[2];
  cv::Mat rgbimg = cv::imread( rgbp, CV_LOAD_IMAGE_COLOR ),
          depthimg = cv::imread( depthp, CV_LOAD_IMAGE_ANYDEPTH );
  float param[4] = { 319.5, 239.5, 525.0, 525.0 };

  // convert to point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGB>() );
  cloud = img2cloud( rgbimg, depthimg, param );

  pcl::visualization::PCLVisualizer::Ptr viewer( new pcl::visualization::PCLVisualizer("viewer") );
  viewer->setBackgroundColor( 0, 0, 0 );
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud" );
  viewer->addCoordinateSystem( 1.0 );
  viewer->initCameraParameters();
  while (!viewer->wasStopped ()) {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  iss3d(cloud);

  harris6d( cloud );

  harris3d( cloud, pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::HARRIS );
  harris3d( cloud, pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::TOMASI );
  harris3d( cloud, pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::LOWE );
  harris3d( cloud, pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::NOBLE );
  harris3d( cloud, pcl::HarrisKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZI>::CURVATURE );

  return 1;
}
