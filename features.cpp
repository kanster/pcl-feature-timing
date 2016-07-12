/**
 * time consumption comparsion analysis for 3D features in Point Cloud Library
 * use uniform sampling as keypoint detector
 * @author Kanzhi Wu
 */


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/rsd.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/rift.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/point_types_conversion.h>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/uniform_sampling.h>

#include <iostream>
#include <string>


/// print command usage
/// argv        -- input arguments
void print_usage( char * argv ) {
  pcl::console::print_error( "Syntax is: %s rgb_image depth_image\n", argv );
//  std::cout << "Options:\n" <<
//               "    -all          use all features\n" <<
//               "    -normal       normal feature\n" <<
//               "    -normalomp    normal feature in OMP mode\n" <<
//               "    -normalii     normal feature using intergral image\n" <<
//               "    -help         print usage\n";
//  std::cout << "\n\n";
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


/// normal vector extraction
///
/// cloud       -- input point cloud
/// kpts        -- input keypoints
///
pcl::PointCloud<pcl::Normal>::Ptr normal_extraction( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud( kpts );
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree( new pcl::search::KdTree<pcl::PointXYZRGB>() );
  ne.setSearchMethod( tree );
  ne.setSearchSurface( cloud );
  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal>() );
  ne.setRadiusSearch( 0.10 );
  ne.compute( *normals );
  double t = tt.toc();
  pcl::console::print_value( "Normal extraction takes %.3f\n", t );
  return normals;
}

/// normal vector extraction in omp
///
/// cloud       -- input point cloud
/// kpts        -- input keypoints
///
pcl::PointCloud<pcl::Normal>::Ptr normal_extraction_omp( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud( kpts );
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree( new pcl::search::KdTree<pcl::PointXYZRGB>() );
  ne.setSearchMethod( tree );
  ne.setSearchSurface( cloud );
  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal>() );
  ne.setRadiusSearch( 0.10 );
  ne.compute( *normals );
  double t = tt.toc();
  pcl::console::print_value( "Normal extraction in OMP takes %.3f\n", t );
  return normals;
}

/// normal vector extraction using integral image
///
/// cloud       -- input point cloud
///
pcl::PointCloud<pcl::Normal>::Ptr normal_extraction_integral_image(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
  ne.setMaxDepthChangeFactor(0.03f);
//  ne.setNormalSmoothingSize(10.0f);
  ne.setInputCloud(cloud);
  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal>() );
  ne.compute( *normals );
  double t = tt.toc();
  pcl::console::print_value( "Normal extraction using integral image(%d points) takes %.3f\n", (int)normals->size(), t );
  return normals;
}

/// Persistent Feature Histogram
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
///
pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::PointCloud<pcl::PointXYZ>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::copyPointCloud( *rgb_cloud, *cloud );
  pcl::copyPointCloud( *rgb_kpts, *kpts );
  pcl::console::TicToc tt;
  tt.tic();
  pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh_extraction;
  pfh_extraction.setSearchSurface( cloud );
  pfh_extraction.setInputCloud( kpts );
  pfh_extraction.setInputNormals( normals );
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pfh_extraction.setSearchMethod( tree );
  pfh_extraction.setRadiusSearch( 0.05 );
  pcl::PointCloud<pcl::PFHSignature125>::Ptr descrs( new pcl::PointCloud<pcl::PFHSignature125>() );
  pfh_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Persistent Feature Histogram takes %.3f\n", t );
  return descrs;
}


/// Persistent Feature Histogram
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr pfhrgb_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ){
  pcl::console::TicToc tt;
  tt.tic();
  pcl::PFHRGBEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHRGBSignature250> pfh_extraction;
  pfh_extraction.setSearchSurface( cloud );
  pfh_extraction.setInputCloud( kpts );
  pfh_extraction.setInputNormals( normals );
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  pfh_extraction.setSearchMethod( tree );
  pfh_extraction.setRadiusSearch( 0.05 );
  pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr descrs( new pcl::PointCloud<pcl::PFHRGBSignature250>() );
  pfh_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Persistent Feature Histogram RGB takes %.3f\n", t );
  return descrs;
}

/// Fast Persistent Feature Histogram
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33 > fpfh_extraction;
  fpfh_extraction.setSearchSurface( cloud );
  fpfh_extraction.setInputCloud( kpts );
  fpfh_extraction.setInputNormals( normals );
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  fpfh_extraction.setSearchMethod( tree );
  fpfh_extraction.setRadiusSearch( 0.05 );
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descrs( new pcl::PointCloud<pcl::FPFHSignature33>() );
  fpfh_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Fast Persistent Feature Histogram takes %.3f\n", t );
  return descrs;
}


/// Fast Persistent Feature Histogram in OMP
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_extraction_omp(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  pcl::console::TicToc tt;
  tt.tic();
  pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33 > fpfh_extraction;
  fpfh_extraction.setSearchSurface( cloud );
  fpfh_extraction.setInputCloud( kpts );
  fpfh_extraction.setInputNormals( normals );
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  fpfh_extraction.setSearchMethod( tree );
  fpfh_extraction.setRadiusSearch( 0.05 );
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descrs( new pcl::PointCloud<pcl::FPFHSignature33>() );
  fpfh_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Fast Persistent Feature Histogram in OMP takes %.3f\n", t );
  return descrs;
}


/// Signature of Hitograms of OrientTation, XYZ
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::SHOT352>::Ptr shot_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  // convert point cloud type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::PointCloud<pcl::PointXYZ>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::copyPointCloud( *rgb_cloud, *cloud );
  pcl::copyPointCloud( *rgb_kpts, *kpts );

  pcl::console::TicToc tt;
  tt.tic();

  pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_extraction;
  shot_extraction.setInputCloud( kpts );
  shot_extraction.setSearchSurface( cloud );
  shot_extraction.setInputNormals( normals );
  shot_extraction.setRadiusSearch( 0.05 );

  pcl::PointCloud<pcl::SHOT352>::Ptr descrs( new pcl::PointCloud<pcl::SHOT352>() );
  shot_extraction.compute( *descrs );

  double t = tt.toc();
  pcl::console::print_value( "Signature of Hitograms of OrientTation takes %.3f\n", t );

  return descrs;
}


/// Signature of Hitograms of OrientTation, XYZ in OMP
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::SHOT352>::Ptr shot_extraction_omp(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  // convert point cloud type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::PointCloud<pcl::PointXYZ>::Ptr kpts( new pcl::PointCloud<pcl::PointXYZ>() );
  pcl::copyPointCloud( *rgb_cloud, *cloud );
  pcl::copyPointCloud( *rgb_kpts, *kpts );

  pcl::console::TicToc tt;
  tt.tic();

  pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_extraction;
  shot_extraction.setInputCloud( kpts );
  shot_extraction.setSearchSurface( cloud );
  shot_extraction.setInputNormals( normals );
  shot_extraction.setRadiusSearch( 0.05 );

  pcl::PointCloud<pcl::SHOT352>::Ptr descrs( new pcl::PointCloud<pcl::SHOT352>() );
  shot_extraction.compute( *descrs );

  double t = tt.toc();
  pcl::console::print_value( "Signature of Hitograms of OrientTation in OMP takes %.3f\n", t );

  return descrs;
}


/// Signature of Hitograms of OrientTation, XYZRGB
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::SHOT1344>::Ptr cshot_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  pcl::console::TicToc tt;
  tt.tic();

  pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot_extraction;
  shot_extraction.setInputCloud( kpts );
  shot_extraction.setSearchSurface( cloud );
  shot_extraction.setInputNormals( normals );
  shot_extraction.setRadiusSearch( 0.05 );

  pcl::PointCloud<pcl::SHOT1344>::Ptr descrs( new pcl::PointCloud<pcl::SHOT1344>() );
  shot_extraction.compute( *descrs );

  double t = tt.toc();
  pcl::console::print_value( "Color Signature of Hitograms of OrientTation takes %.3f\n", t );

  return descrs;
}


/// Signature of Hitograms of OrientTation, XYZRGB in OMP
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::SHOT1344>::Ptr cshot_extraction_omp(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  pcl::console::TicToc tt;
  tt.tic();

  pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot_extraction;
  shot_extraction.setInputCloud( kpts );
  shot_extraction.setSearchSurface( cloud );
  shot_extraction.setInputNormals( normals );
  shot_extraction.setRadiusSearch( 0.05 );

  pcl::PointCloud<pcl::SHOT1344>::Ptr descrs( new pcl::PointCloud<pcl::SHOT1344>() );
  shot_extraction.compute( *descrs );

  double t = tt.toc();
  pcl::console::print_value( "Color Signature of Hitograms of OrientTation in OMP takes %.3f\n", t );

  return descrs;
}


/// Radius-Based Surface Descriptor
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr rsd_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {
  pcl::console::TicToc tt;
  tt.tic();

  pcl::RSDEstimation< pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalRadiiRSD > rsd_extraction;
  rsd_extraction.setInputCloud( kpts );
  rsd_extraction.setSearchSurface(cloud);
  rsd_extraction.setInputNormals( normals );

  rsd_extraction.setRadiusSearch( 0.05 );
  rsd_extraction.setPlaneRadius( 0.1 );
  rsd_extraction.setSaveHistograms( false );

  pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr descrs( new pcl::PointCloud<pcl::PrincipalRadiiRSD>() );
  rsd_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Radius-Based Surface Descriptor takes %.3f\n", t );

  return descrs;
}

/// 3D Shape Context
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
pcl::PointCloud<pcl::ShapeContext1980>::Ptr sc_extraction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts,
    pcl::PointCloud<pcl::Normal>::Ptr normals ) {

  pcl::console::TicToc tt;
  tt.tic();

  pcl::ShapeContext3DEstimation< pcl::PointXYZRGB, pcl::Normal, pcl::ShapeContext1980 > sc_extraction;
  sc_extraction.setInputCloud( kpts );
  sc_extraction.setSearchSurface(cloud);
  sc_extraction.setInputNormals( normals );

  sc_extraction.setRadiusSearch( 0.05 );
  sc_extraction.setMinimalRadius(0.05 / 10.0);
  sc_extraction.setPointDensityRadius(0.05 / 5.0);

  pcl::PointCloud<pcl::ShapeContext1980>::Ptr descrs( new pcl::PointCloud<pcl::ShapeContext1980>() );
  sc_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "3D Shape Context takes %.3f\n", t );

  return descrs;

}


/// Rotation-Invariant Feature Transform
///
/// cloud         -- input point cloud
/// kpts          -- keypoints
/// normals       -- normals
///
typedef pcl::Histogram<32> RIFT32;
pcl::PointCloud<RIFT32 >::Ptr rift_extraction( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr kpts, pcl::PointCloud<pcl::Normal>::Ptr normals ) {

  pcl::console::TicToc tt;
  tt.tic();

  // Convert the RGB to intensity.
  pcl::PointCloud<pcl::PointXYZI>::Ptr intensity_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloudXYZRGBtoXYZI(*cloud, *intensity_cloud);
  pcl::PointCloud<pcl::PointXYZI>::Ptr intensity_kpts(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloudXYZRGBtoXYZI(*kpts, *intensity_kpts);

  // Compute the intensity gradients.
  pcl::PointCloud<pcl::IntensityGradient>::Ptr gradients(new pcl::PointCloud<pcl::IntensityGradient>);
  pcl::IntensityGradientEstimation< pcl::PointXYZI, pcl::Normal, pcl::IntensityGradient,
      pcl::common::IntensityFieldAccessor<pcl::PointXYZI> > ge;
  ge.setInputCloud( intensity_kpts );
  ge.setSearchSurface( intensity_cloud );
  ge.setInputNormals(normals);
  ge.setRadiusSearch(0.05);
  ge.compute(*gradients);
  pcl::console::print_value( "gradients = %d, keypoints = %d\n", (int)gradients->size(), (int)intensity_kpts->size() );

  pcl::RIFTEstimation<pcl::PointXYZI, pcl::IntensityGradient, RIFT32 > rift_extraction;
  rift_extraction.setInputCloud(intensity_kpts);
//  rift_extraction.setSearchSurface( intensity_cloud );
  rift_extraction.setInputGradient(gradients);
  rift_extraction.setRadiusSearch(0.05);
  rift_extraction.setNrDistanceBins(4);
  rift_extraction.setNrGradientBins(8);


  pcl::PointCloud<RIFT32 >::Ptr descrs( new pcl::PointCloud<RIFT32 >() );
  rift_extraction.compute( *descrs );
  double t = tt.toc();
  pcl::console::print_value( "Rotation-Invariant Feature Transform takes %.3f\n", t );

  return descrs;

}

int main( int argc, char ** argv ) {
  pcl::console::setVerbosityLevel( pcl::console::L_ERROR );

  if ( argc < 2 ) {
    print_usage( argv[0] );
    return (-1);
  }

  // argument parser
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

  // keypoint extraction
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints( new pcl::PointCloud<pcl::PointXYZRGB>() );
  pcl::UniformSampling<pcl::PointXYZRGB> uniform_sampling;
  uniform_sampling.setInputCloud( cloud );
  uniform_sampling.setRadiusSearch( 0.05 );
  uniform_sampling.filter( *keypoints );
  pcl::console::print_value( "Extract %d keypoints from uniform sampling\n", (int)keypoints->size() );


  // normal extraction
  normal_extraction( cloud, keypoints );

  // normal extraction in omp
  normal_extraction_omp( cloud, keypoints );

  // normal extraction using integral image
  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal>() );
  normals = normal_extraction_integral_image( cloud );

  // RIFT feature
  rift_extraction( cloud, keypoints, normals );

  // 3DSC feature
  sc_extraction( cloud, keypoints, normals );

  // RSD feature
  rsd_extraction( cloud, keypoints, normals );

  // SHOT feature
  shot_extraction( cloud, keypoints, normals );
  shot_extraction_omp( cloud, keypoints, normals );
  cshot_extraction( cloud, keypoints, normals );
  cshot_extraction_omp( cloud, keypoints, normals );

  // FPFH feature
  fpfh_extraction( cloud, keypoints, normals );
  fpfh_extraction_omp( cloud, keypoints, normals );

  // PFH feature
  pfh_extraction( cloud, keypoints, normals );

  // PFHRGB feature
  pfhrgb_extraction( cloud, keypoints, normals );

  return (1);
}
