#include<iostream>
#include<sstream>
#include<string>
#include<vector>

#include"opencv2/opencv.hpp"
#include"opencv2/cudastereo.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include"FlyCap2CV.h"

using namespace std;

/*
static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}
*/


static void saveXYZ_PCD(const char* filename, const cv::Mat& mat)
{
  int n=0;
  const double max_z = 3.0e1;
  FILE* fp = fopen(filename, "wt");
  fprintf(fp, "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH        \nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS        \nDATA ascii\n");

  for(int y = 0; y < mat.rows; y++)
    {
      for(int x = 0; x < mat.cols; x++)
        {
	  cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
	  if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
	    continue;
	  fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
	  n++;
        }
    }

    fseek(fp, 65, SEEK_SET);
    fprintf(fp, "%7d", n);
    fseek(fp, 113, SEEK_SET);
    fprintf(fp, "%7d", n);
    
    fclose(fp);
}


int stereo_match(int argc, char* argv[])
{
  int l, r;

  cout << "which is left? (0 or 1) = ";
  cin >> l;
  cout << endl;

  if(l == 0)
    r = 1;
  else if(l == 1)
    r = 0;
  
  FlyCap2CVWrapper cam0(l);
  FlyCap2CVWrapper cam1(r);
  
  //cout << cam0.getCameraSN() << endl;
  //cout << cam1.getCameraSN() << endl;
  
  cv::FileStorage fs;
  fs.open("../data/stereo_extrinsics.yml", cv::FileStorage::READ);
  cv::Mat K0, D0, K1, D1, R, T, E, F;
  cv::Rect roi1, roi2;
  
  fs["K0"] >> K0;
  fs["D0"] >> D0;
  fs["K1"] >> K1;
  fs["D1"] >> D1;
  fs["R"] >> R;
  fs["T"] >> T;
  fs["E"] >> E;
  fs["F"] >> F;

  cout << endl;
  cout << "K0 " << K0 << endl;
  cout << "K1 " << K1 << endl;
  cout << "D0 " << D0 << endl;
  cout << "D1 " << D1 << endl;
  cout << "E " << E << endl;
  cout << "F " << F << endl;
  cout << "R " << R << endl;
  cout << "T " << T << endl;
  
  cv::Mat frame0, frame1;
  cv::Mat disp, disp8;
  cv::Mat gray0, gray1;
  cv::Mat dst;
  
  frame0 = cam0.readImage();
  if(frame0.empty())
    {
      cout << "Can't get image from cam." << endl;
      return 0;
    }
  
  frame1 = cam1.readImage();
  if(frame1.empty())
    {
      cout << "Can't get image from cam." << endl;
      return 0;
    }
   
  cv::Mat combine(cv::Size(frame0.cols*2, frame0.rows), CV_8UC3);
  cv::Mat combines;
  cv::Mat xyz;
  
  cv::Mat R0, R1, P0, P1, Q;
  cv::stereoRectify(K0, D0, K1, D1, frame0.size(), R, T, R0, R1, P0, P1, Q,
		    cv::CALIB_ZERO_DISPARITY);

  cv::Mat map01, map02, map11, map12;
  cv::initUndistortRectifyMap(K0, D0, R0, P0, frame0.size(), CV_32FC1, map01,
			      map02);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, frame1.size(), CV_32FC1, map11,
			      map12);

  cout << "Q " << Q << endl;

  // CUDA StereoBM -----------
  /*
  cv::Ptr<cv::cuda::StereoBM> sm = cv::cuda::createStereoBM(256,3);
  cv::Ptr<cv::cuda::DisparityBilateralFilter> dbf = cv::cuda::createDisparityBilateralFilter(256,3,1);
  //  cv::Ptr<cv::cuda::StereoBeliefPropagation> sm = cv::cuda::createStereoBeliefPropagation(256);
  //  cv::Ptr<cv::cuda::StereoConstantSpaceBP> sm = cv::cuda::createStereoConstantSpaceBP(256);
  */

  // StereoBM -----------
  //cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(80, 21);

  // StereoSGBM -----------
  cv::Ptr<cv::StereoSGBM> sgbm =cv::StereoSGBM::create(4, 128, 11, 100, 1000,
				32, 0, 15, 1000, 16, cv::StereoSGBM::MODE_SGBM);

  // KAZE -----------
  //vector<cv::KeyPoint> kpts0, kpts1;
  //cv::Mat desc0, desc1;
  
  //cv::Ptr<cv::KAZE> detector = cv::KAZE::create();
  //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  //vector<cv::DMatch> matches;

  int n=100;
  string dnum;

  cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Image", 300, 50);
  cv::namedWindow("Disp", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Disp", 500, 400);
  
  while(true)
    {
      frame0 = cam0.readImage();
      if(frame0.empty())
	{
	  cout << "Can't get image from cam." << endl;
	  break;
	}
      
      frame1 = cam1.readImage();
      if(frame1.empty())
	{
	  cout << "Can't get image from cam." << endl;
	  break;
	}
      
      cv::remap(frame0, frame0, map01, map02, cv::INTER_LINEAR);
      cv::remap(frame1, frame1, map11, map12, cv::INTER_LINEAR);
      
      cv::cvtColor(frame0, gray0, cv::COLOR_BGR2GRAY);
      cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);

      // CUDA StereoBM -----------
      /*
      cv::cuda::GpuMat d_left,d_right;
      d_left.upload(gray0);
      d_right.upload(gray1);
      cv::cuda::GpuMat d_disp(frame0.size(), CV_8UC1);

      sm->compute(d_left,d_right,d_disp);
      dbf->apply(d_disp,d_left,d_disp);
      d_disp.download(disp);
      */
      
      //disp.convertTo(disp,CV_8U,255.0/(256*16.0));

      // StereoBM -----------
      //sbm->compute(gray0, gray1, disp);
      //disp.convertTo(disp8, CV_8UC1, 1/16.0);

      // StereoSGBM ----------
      sgbm->compute(gray0, gray1, disp);
      disp.convertTo(disp8, CV_8UC1, 1/16.0);

      // KAZE -----------
      //detector->detectAndCompute(frame0, cv::noArray(), kpts0, desc0);
      //detector->detectAndCompute(frame1, cv::noArray(), kpts1, desc1);

      //matcher->match(desc0, desc1, matches);
      //cv::drawMatches(frame0, kpts0, frame1, kpts1, matches, dst);
      //cv::imshow("dst", dst);

      //cout << kpts0.at(0) << endl;
      
      cv::Mat imageLeft(combine, cv::Rect(0, 0, frame0.cols, frame0.rows));
      cv::Mat imageRight(combine,
			 cv::Rect(frame0.cols, 0, frame1.cols, frame1.rows));
      frame0.copyTo(imageLeft);
      frame1.copyTo(imageRight);

      cv::resize(combine, combines, cv::Size(0, 0), 0.5, 0.5);
    
      n++;
      dnum=to_string(n);
    
      cv::imshow("Image", combines);
      //cv::imwrite("../data/SV"+dnum+".ppm", combine);
      cv::imwrite("../data/SV.ppm", combine);

      cv::imshow("Disp", disp8);
      //cv::imwrite("../data/dips"+dnum+".pgm", disp);
      cv::imwrite("../data/dips.pgm", disp8);
    
      cv::reprojectImageTo3D(disp8, xyz, Q, true);
      saveXYZ_PCD("../data/pl.pcd", xyz);
      //saveXYZ_PCD("../data/pl"+dnum+".pcd", xyz);
    
      int key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
    }

  cv::destroyWindow("Image");
  cv::destroyWindow("Disp");
  return 0;
}


int stereo_record(int argc,char* argv[])
{
  int l, r, t;

  cout << "which is left? (0 or 1) = ";
  cin >> l;
  cout << endl;
   
  cout << "Input interval (ms) = ";
  cin >> t;
   cout << endl;
   
   if(l == 0)
     r = 1;
   else if(l == 1)
     r = 0;
  
   FlyCap2CVWrapper cam0(l);
   FlyCap2CVWrapper cam1(r);
  
   cv::FileStorage fs;
   fs.open("../data/stereo_extrinsics.yml", cv::FileStorage::READ);
   cv::Mat K0, D0, K1, D1, R, T, E, F;
   cv::Rect roi1, roi2;
  
   fs["K0"] >> K0;
   fs["D0"] >> D0;
   fs["K1"] >> K1;
   fs["D1"] >> D1;
   fs["R"] >> R;
   fs["T"] >> T;
   fs["E"] >> E;
   fs["F"] >> F;

   cout << endl;
   cout << "K0 " << K0 << endl;
   cout << "K1 " << K1 << endl;
   cout << "D0 " << D0 << endl;
   cout << "D1 " << D1 << endl;
   cout << "E " << E << endl;
   cout << "F " << F << endl;
   cout << "R " << R << endl;
   cout << "T " << T << endl;
  
   cv::Mat frame0, frame1;
   cv::Mat disp, disp8;
   cv::Mat gray0, gray1; //, dis0, dis1;
   cv::Mat dst;
  
   frame0 = cam0.readImage();
   if(frame0.empty())
     {
       cout << "Can't get image from cam0." << endl;
       return 0;
     }
   
   frame1 = cam1.readImage();
   if(frame1.empty())
     {
       cout << "Can't get image from cam1." << endl;
       return 0;
     }
  
   //cv::cvtColor(frame0, gray0, cv::COLOR_BGR2GRAY);
   //cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
  
   cv::Mat combine(cv::Size(frame0.cols*2, frame0.rows), CV_8UC3);
   cv::Mat combines;
  
   cv::Mat R0, R1, P0, P1, Q;
   cv::stereoRectify(K0, D0, K1, D1, frame0.size(), R, T, R0, R1, P0, P1, Q,
		     cv::CALIB_ZERO_DISPARITY);

   cv::Mat map01, map02, map11, map12;
   cv::initUndistortRectifyMap(K0, D0, R0, P0, frame0.size(), CV_32FC1, map01,
			       map02);
   cv::initUndistortRectifyMap(K1, D1, R1, P1, frame1.size(), CV_32FC1, map11,
			       map12);

   cout << "Q " << Q << endl;

   int n=100;
   string dnum;

   cv::namedWindow("Stereo Camera", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Stereo Camera", 300, 50);
  
   while(true)
     {
       frame0 = cam0.readImage();
       if(frame0.empty())
	 {
	   cout << "Can't get image from cam." << endl;
	   break;
	 }
       
       frame1 = cam1.readImage();
       if(frame1.empty())
	 {
	   cout << "Can't get image from cam." << endl;
	   break;
	 }
       
       cv::remap(frame0, frame0, map01, map02, cv::INTER_LINEAR);
       cv::remap(frame1, frame1, map11, map12, cv::INTER_LINEAR);
         
       cv::Mat imageLeft(combine, cv::Rect(0, 0, frame0.cols, frame0.rows));
       cv::Mat imageRight(combine, cv::Rect(frame0.cols, 0, frame1.cols,
					    frame1.rows));
       frame0.copyTo(imageLeft);
       frame1.copyTo(imageRight);

       cv::resize(combine, combines, cv::Size(0, 0), 0.75, 0.75);
    
       n++;
       dnum=to_string(n);
    
       //imshow("combined", combine);
       cv::imshow("Stereo Camera", combines);
       //cv::imwrite("../data/SV"+dnum+".ppm", combine);

       int key = cv::waitKey(t);
       if(key == 'q')
	 break;
       if(key == 27)
	 break;
     }

   cv::destroyWindow("Stereo Camera");
   return 0;
}


int main(int argc, char* argv[])
{
  char f;
  
  cout << endl;
  cout << "*** Stereo Matching ***" << endl;
  
  while(true)
    {
      cout << endl;
      cout << "Please Select" << endl;
      cout << " Stereo Vision (s)" << endl;
      cout << " Image Record (r)" << endl;
      cout << " Quit (q)" << endl;
      cout << "Which ? = ";
      cin >> f;
      cout << endl;
  
      if(f == 's')
	stereo_match(argc, argv);
  
      else if(f == 'r')
	stereo_record(argc, argv);
  
      else if(f == 'q')
	break;
    }

  return 0;
}

