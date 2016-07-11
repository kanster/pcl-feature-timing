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
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <iostream>



void print_usage( char * argv ) {
  pcl::console::print_error( "Syntax is: %s rgb_image depth_image <options>\n", argv );
  std::cout << "Options:\n" <<
               "    -all          use all features\n" <<
               "    -normal       normal feature\n" <<
               "    -normalomp    normal feature in OMP mode\n" <<
               "    -help         print usage\n";
  std::cout << "\n\n";
}




int main( int argc, char ** argv ) {
  if ( argc < 2 ) {
    print_usage( argv[0] );
    return (-1);
  }

  if (pcl::console::find_switch( argc, argv, "-help" )) {
    print_usage( argv[0] );
  }

  return 1;
}
