#include<iostream>
#include<sstream>
#include<string>
#include<vector>

#include"opencv2/opencv.hpp"
#include"opencv2/cudastereo.hpp"
#include"FlyCap2CV.h"

using namespace std;
using namespace cv;

//Chessboard info
float SQ_SIZE = 0.04;                       // meter
int CH=6;
int CW=8;

//Camera info
float PS=0.0053;
float F=16.0;
float IW=1280.0;
float IH=960.0;


int check_view(int argc,char* argv[])
{
  FlyCap2CVWrapper cam0(0);
  FlyCap2CVWrapper cam1(1);
  
  //cout << cam0.getCameraSN() << endl;
  //cout << cam1.getCameraSN() << endl;

  cv::Mat frame0, frame1, gray0, gray1;
  cv::Mat hist0, hist1;
  
  float v_ranges[]={0, 255};
  const float* ranges[]={v_ranges};
  int histSize[]={256};
  int ch[]={0};
  
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
  
  cv::Mat combine(cv::Size(frame0.cols*2, frame0.rows), CV_8UC3);
  cv::Mat combines;

  cv::namedWindow("Cam View", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Cam View", 100, 100);
  cv::namedWindow("Hist", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Hist", 300, 850);
  
  while(true)
    {
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

      cv::cvtColor(frame0, gray0, cv::COLOR_BGR2GRAY);
      cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);

      cv::Mat histg(300, 580, CV_8UC1, cv::Scalar::all(255));
      //double maxVal=0;
      //cv::Point maxLoc;
      
      cv::calcHist(&gray0, 1, ch, cv::noArray(), hist0, 1, histSize, ranges, true);
      cv::calcHist(&gray1, 1, ch, cv::noArray(), hist1, 1, histSize, ranges, true);
      //cv::minMaxLoc(hist0, 0, &maxVal, 0, &maxLoc);
      cv::normalize(hist0, hist0, 0, 255, cv::NORM_MINMAX);
      cv::normalize(hist1, hist1, 0, 255, cv::NORM_MINMAX);
      
      cv::rectangle(histg, cv::Point(285, 0), cv::Point(295, 299),
		    cv::Scalar(30, 30, 30), -1, cv::LINE_AA, 0);
      
      for(int i=0; i<256; i++)
	{
	  cv::line(histg, cv::Point(i+5, 299),
		   cv::Point(i+5, 299-hist0.at<float>(cv::Point(0, i))),
		   cv::Scalar(0, 0, 0), 1, 0);
	  cv::line(histg, cv::Point(i+300, 299),
		   cv::Point(i+300, 299-hist1.at<float>(cv::Point(0, i))),
		   cv::Scalar(0, 0, 0), 1, 0);
	}
	
      cv::Mat imageLeft(combine, cv::Rect(0, 0, frame0.cols, frame0.rows));
      cv::Mat imageRight(combine,
			 cv::Rect(frame0.cols, 0, frame1.cols, frame1.rows));
      frame0.copyTo(imageLeft);
      frame1.copyTo(imageRight);

      cv::line(combine, cv::Point(0, frame0.rows*0.5),
	       cv::Point(frame0.cols*2, frame0.rows*0.5),
	       cv::Scalar(0, 200, 0), 1, 0);
      cv::line(combine, cv::Point(frame0.cols*0.5, 0),
	       cv::Point(frame0.cols*0.5, frame0.rows), cv::Scalar(0, 200, 0), 1, 0);
      cv::line(combine, cv::Point(frame0.cols*1.5, 0),
	       cv::Point(frame0.cols*1.5, frame0.rows), cv::Scalar(0, 200, 0), 1, 0);
      cv::putText(combine, "Cam 0 (SN: "+cam0.getCameraSN()+")",
		  cv::Point(10, 50), cv::FONT_HERSHEY_TRIPLEX, 1.2,
		  cv::Scalar(200, 100, 0), 1, cv::LINE_AA);
      cv::putText(combine, "Cam 1 (SN: "+cam1.getCameraSN()+")",
		  cv::Point(frame0.cols+10, 50), cv::FONT_HERSHEY_TRIPLEX, 1.2,
		  cv::Scalar(200, 100, 0), 1, cv::LINE_AA);
      
      cv::resize(combine, combines, cv::Size(0, 0), 0.75, 0.75);
      cv::imshow("Cam View", combines);
      cv::imshow("Hist", histg);

      int key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
    }

  cv::destroyWindow("Cam View");
  cv::destroyWindow("Hist");
  
  return 0;
}


int check_stereo(int argc, char* argv[])
{
  int l, r;

  cout << "whitch is left? (0 or 1) = ";
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
  fs.open("../data/stereo_extrinsics.yml",FileStorage::READ);
  Mat K0, D0, K1, D1, R, T, E, F;
  Rect roi1, roi2;
  
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
  cout << "D1 " << D1 << endl << endl;
  cout << "E " << E << endl;
  cout << "F " << F << endl;
  cout << "R " << R << endl;
  cout << "T " << T << endl;
  
  Mat frame0, frame1;
  Mat gray0, gray1;
  vector<Point2f> corner0, corner1;
  Mat draw0, draw1;
  
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
        
  Mat combine(Size(frame0.cols*2, frame0.rows), CV_8UC3);
  Mat combines;
  
  Mat R0, R1, P0, P1, Q;
  cv::stereoRectify(K0, D0, K1, D1, frame0.size(), R, T, R0, R1, P0, P1, Q,
		    CALIB_ZERO_DISPARITY);
  Mat map01, map02, map11, map12;
  //initUndistortRectifyMap(K0,D0,R,K0,frame0.size(),CV_32FC1,map01,map02);
  //initUndistortRectifyMap(K1,D1,R,K1,frame1.size(),CV_32FC1,map11,map12);
  cv::initUndistortRectifyMap(K0, D0, R0, P0, frame0.size(), CV_32FC1, map01, map02);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, frame1.size(), CV_32FC1, map11, map12);

  cout << "Q " << Q << endl << endl;

  while(true)
    {
      frame0 = cam0.readImage();
      if(frame0.empty())
	{
	  cout << "Can't get image from cam0." << endl;
	  return 0;
    }
      
      frame1 = cam1.readImage();
      
      cv::remap(frame0, frame0, map01, map02, INTER_LINEAR);
      cv::remap(frame1, frame1, map11, map12, INTER_LINEAR);
      
      cv::cvtColor(frame0, gray0, COLOR_BGR2GRAY);
      cvtColor(frame1, gray1, COLOR_BGR2GRAY);

      bool found0 = cv::findChessboardCorners(gray0, cv::Size(CW, CH), corner0,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
      bool found1 = cv::findChessboardCorners(gray1,cv::Size(CW, CH), corner1,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
           
      if(found0)
	{
	  cv::cornerSubPix(gray0, corner0, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));
	}
      if(found1)
	{
	  cv::cornerSubPix(gray1, corner1, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));
	}
            
      frame0.copyTo(draw0);
      cv::drawChessboardCorners(draw0, cv::Size(CW,CH), cv::Mat(corner0), found0);
      frame1.copyTo(draw1);
      cv::drawChessboardCorners(draw1, cv::Size(CW,CH), cv::Mat(corner1), found1);

      if(found0 && found1)
	{
	  cv::line(draw0, cv::Point(0, cvRound(corner0.at(0).y)), cv::Point(frame0.cols, cvRound(corner0.at(0).y)), Scalar(0, 200, 0), 1, 0);
	  cv::line(draw1, cv::Point(0, cvRound(corner1.at(0).y)), cv::Point(frame1.cols, cvRound(corner1.at(0).y)), Scalar(0, 200, 0), 1, 0);
	  cv::line(draw0, cv::Point(0, cvRound(corner0.at(20).y)), cv::Point(frame0.cols, cvRound(corner0.at(20).y)), Scalar(0, 200, 200), 1, 0);
	  cv::line(draw1, cv::Point(0, cvRound(corner1.at(20).y)), cv::Point(frame1.cols, cvRound(corner1.at(20).y)), Scalar(0, 200, 200), 1, 0);
	  cv::line(draw0, cv::Point(0, cvRound(corner0.at(47).y)), cv::Point(frame0.cols, cvRound(corner0.at(47).y)), Scalar(200, 200, 0), 1, 0);
	  cv::line(draw1, cv::Point(0, cvRound(corner1.at(47).y)), cv::Point(frame1.cols, cvRound(corner1.at(47).y)), Scalar(200, 200, 0), 1, 0);
	}
      
      Mat imageLeft(combine, Rect(0, 0, frame0.cols, frame0.rows));
      Mat imageRight(combine, Rect(frame0.cols, 0, frame1.cols, frame1.rows));
      draw0.copyTo(imageLeft);
      draw1.copyTo(imageRight);

      cv::resize(combine, combines, Size(0, 0), 0.75, 0.75);
      cv::imshow("Stereo Check", combines);
    
      int key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
      if(key == 's' && found0 && found1)
	{
	  imwrite("../data/SV_test.ppm", combine);
	  //for(int i; i<48; i++)
	  //{
	  cout << corner0.at(0) << endl;
	  cout << corner1.at(0) << endl << endl;
	  cout << corner0.at(20) << endl;
	  cout << corner1.at(20) << endl << endl;
	  cout << corner0.at(47) << endl;
	  cout << corner1.at(47) << endl << endl;
	  //}
	}
    }

  cv::destroyWindow("Stereo Check");
  return 0;
}


int check_mono(int argc, char* argv[])
{
  int cam_num;
  cout << "Camera Number? (0 or 1) = ";
  cin >> cam_num;
  cout << endl;

  FlyCap2CVWrapper cam(cam_num);
  cv::Mat frame, dst;
  
  cv::FileStorage fs;
  cv::Mat K, D;
  
  fs.open("../data/"+cam.getCameraSN()+"_intrinsics.yml", cv::FileStorage::READ);
  fs["K"] >> K;
  fs["D"] >> D;
  
  cout << "K " << K << endl;
  cout << "D " << D << endl << endl;

  fs.release();

  int LN=10;
  
  cv::namedWindow("Mono Check", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Mono Check", 200, 100);
  
  while(true)
    {
      frame = cam.readImage();
      if(frame.empty())
	{
	  cout << "Can't get image from Camera." << endl;
	  break;
	}
      cv::undistort(frame, dst, K, D);
      
      for(int i=1; i<LN; i++)
	{
	  cv::line(dst, cv::Point(0, frame.rows*i/LN),
		   cv::Point(frame.cols, frame.rows*i/LN),
		   cv::Scalar(0, 200, 0), 1, 0);
	  cv::line(dst, cv::Point(frame.cols*i/LN ,0),
		   cv::Point(frame.cols*i/LN, frame.rows),
		   cv::Scalar(0, 200, 0), 1, 0);
	}
    
      cv::imshow("Mono Check", dst);
    
      int key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
    }

  cv::destroyWindow("Mono Check");
  return 0;
}


int stereo_calibrate(int argc,char* argv[])
{
  FlyCap2CVWrapper cam0(0);
  FlyCap2CVWrapper cam1(1);

  FileStorage fs0;
  FileStorage fs1;
  int l;
  
  Mat frame0, frame1;
  Mat gray0, gray1;
  std::vector<cv::Point2f> corner0,corner1;
  std::vector<std::vector<cv::Point2f>> corner0_list,corner1_list;

  frame0 = cam0.readImage();
  if(frame0.empty())
    {
      cout << "Can't get image from Cam0." << endl;
      return 0;
	}

  frame1 = cam1.readImage();
  if(frame1.empty())
    return 0;
  
  Mat combine(Size(frame0.cols*2, frame0.rows), CV_8UC3);
  Mat combines;
  
  Mat draw0, draw1;
  frame0.copyTo(draw0);
  frame1.copyTo(draw1);
  
  cv::Mat imageLeft(combine, cv::Rect(0, 0, draw0.cols, draw0.rows));
  cv::Mat imageRight(combine, cv::Rect(draw0.cols, 0, draw1.cols, draw1.rows));
  draw0.copyTo(imageLeft);
  draw1.copyTo(imageRight);
  
  cv::putText(combine, "Cam 0 (SN: "+cam0.getCameraSN()+")", Point(10, 50), FONT_HERSHEY_TRIPLEX, 1.2, Scalar(200, 100, 0), 1, LINE_AA);
  cv::putText(combine, "Cam 1 (SN: "+cam1.getCameraSN()+")", Point(frame0.cols+10, 50), FONT_HERSHEY_TRIPLEX, 1.2, Scalar(200, 100, 0), 1, LINE_AA);
  cv::resize(combine, combines, Size(0, 0), 0.75, 0.75);
  cv::imshow("Stereo Calibration",combines);
  
  int key = cv::waitKey(1000);
  cout << "Which is left? (0 or 1) = ";
  cin >> l;
  cout << endl;

  if(l == 0)
    {
      fs0.open("../data/"+cam0.getCameraSN()+"_intrinsics.yml", cv::FileStorage::READ);
      fs1.open("../data/"+cam1.getCameraSN()+"_intrinsics.yml", cv::FileStorage::READ);
    }
  else if(l == 1)
    {
      fs0.open("../data/"+cam1.getCameraSN()+"_intrinsics.yml", cv::FileStorage::READ);
      fs1.open("../data/"+cam0.getCameraSN()+"_intrinsics.yml", cv::FileStorage::READ);
    }
  
  Mat K0, D0, K1, D1;
  fs0["K"] >> K0;
  fs0["D"] >> D0;
  fs1["K"] >> K1;
  fs1["D"] >> D1;
  cout << "K0 " << K0 << endl;
  cout << "K1 " << K1 << endl;
  cout << "D0 " << D0 << endl;
  cout << "D1 " << D1 << endl << endl;

  fs0.release();
  fs1.release();
  
  while(true)
    {
      if(l == 0)
	{
	  frame0 = cam0.readImage();
	  frame1 = cam1.readImage();
	}
      else if(l == 1)
	{
	  frame0 = cam1.readImage();
	  frame1 = cam0.readImage();
	}
      
      cv::cvtColor(frame0, gray0, COLOR_BGR2GRAY);
      cv::cvtColor(frame1, gray1, COLOR_BGR2GRAY);

      bool found0 = cv::findChessboardCorners(gray0,cv::Size(CW,CH), corner0,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
      bool found1 = cv::findChessboardCorners(gray1,cv::Size(CW,CH), corner1,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

      if(found0)
	{
	  cv::cornerSubPix(gray0,corner0,cv::Size(11,11),cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,30,0.1));
	}
      if(found1)
	{
	  cv::cornerSubPix(gray1,corner1,cv::Size(11,11),cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,30,0.1));
	}

      frame0.copyTo(draw0);
      cv::drawChessboardCorners(draw0,cv::Size(CW,CH),cv::Mat(corner0),found0);
      frame1.copyTo(draw1);
      cv::drawChessboardCorners(draw1,cv::Size(CW,CH),cv::Mat(corner1),found1);

      cv::Mat imageLeft(combine, cv::Rect(0, 0, draw0.cols, draw0.rows));
      cv::Mat imageRight(combine, cv::Rect(draw0.cols, 0, draw1.cols, draw1.rows));
      draw0.copyTo(imageLeft);
      draw1.copyTo(imageRight);

      if(l == 0)
	{
	  cv::putText(combine, "Cam 0 (SN: "+cam0.getCameraSN()+")",
		      Point(10, 50), FONT_HERSHEY_TRIPLEX, 1.2, Scalar(200, 100, 0), 1, LINE_AA);
	  cv::putText(combine, "Cam 1 (SN: "+cam1.getCameraSN()+")",
		  Point(frame0.cols+10, 50), FONT_HERSHEY_TRIPLEX, 1.2,
		  Scalar(200, 100, 0), 1, LINE_AA);
	}
      else if(l == 1)
	{
	  cv::putText(combine, "Cam 1 (SN: "+cam1.getCameraSN()+")", Point(10, 50), FONT_HERSHEY_TRIPLEX, 1.2, Scalar(200, 100, 0), 1, LINE_AA);
	  cv::putText(combine, "Cam 0 (SN: "+cam0.getCameraSN()+")", Point(frame0.cols+10, 50), FONT_HERSHEY_TRIPLEX, 1.2, Scalar(200, 100, 0), 1, LINE_AA);
	}
      
      cv::resize(combine, combines, Size(0, 0), 0.75, 0.75);
      cv::imshow("Stereo Calibration",combines);
    
      key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
      if(key == 's' && found0 && found1)
	{
	  corner0_list.push_back(corner0);
	  corner1_list.push_back(corner1);
	  cout<< "corner saved:" << corner0_list.size() << endl;
	}
    }
  
  std::vector<cv::Point3f> object;
  for(int j=0; j < CH; j++)
    {
      for(int i=0; i < CW; i++)
	{
      object.push_back(cv::Point3f(i*SQ_SIZE, j*SQ_SIZE, 0.0f));
    }
  }
  
  std::vector<std::vector<cv::Point3f> > object_list;
  for(std::size_t i=0;i<corner0_list.size();i++){
    object_list.push_back(object);
  }
  Mat R, T, E, F;
  
  cv::stereoCalibrate(object_list, corner0_list, corner1_list, K0, D0, K1, D1,
		      frame0.size(), R, T, E, F,
		      CALIB_FIX_INTRINSIC
		      //CALIB_USE_INTRINSIC_GUESS
		      );
  /*
    if(l == 0)
      {
  fs0.open("../data/"+cam0.getCameraSN()+"_intrinsics.yml",cv::FileStorage::WRITE);
  fs1.open("../data/"+cam1.getCameraSN()+"_intrinsics.yml",cv::FileStorage::WRITE);
      }
    else if(l ==1)
      {
	fs0.open("../data/"+cam1.getCameraSN()+"_intrinsics.yml",cv::FileStorage::WRITE);
	fs1.open("../data/"+cam0.getCameraSN()+"_intrinsics.yml",cv::FileStorage::WRITE);
      }

  fs0<<"SN"<<cam0.getCameraSN();
  fs0<<"K"<<K0;
  fs0<<"D"<<D0;
  fs1<<"SN"<<cam1.getCameraSN();
  fs1<<"K"<<K1;
  fs1<<"D"<<D1;

  fs0.release();
  fs1.release();
  */
  cv::FileStorage fs("../data/stereo_extrinsics.yml",cv::FileStorage::WRITE);
  if(l == 0)
    {
  fs<<"left SN"<<cam0.getCameraSN();
  fs<<"right SN"<<cam1.getCameraSN();
    }
  else if(l == 1)
    {
      fs<<"left SN"<<cam1.getCameraSN();
      fs<<"right SN"<<cam0.getCameraSN();
    }
  
  fs<<"K0"<<K0;
  fs<<"D0"<<D0;
  fs<<"K1"<<K1;
  fs<<"D1"<<D1;  
  fs<<"R"<<R;
  fs<<"T"<<T;
  fs<<"E"<<E;
  fs<<"F"<<F;
  fs.release();

  cout << endl;
  cout << "K0 " << K0 << endl;
  cout << "K1 " << K1 << endl;
  cout << "D0 " << D0 << endl;
  cout << "D1 " << D1 << endl;
  cout << "R " << R << endl;
  cout << "T " << T << endl;
  cout << "E " << E << endl;
  cout << "F " << F << endl;

  cv::destroyWindow("Stereo Calibration");
  return 0;
}


int mono_calibrate(int argc, char* argv[])
{
  int cam_num;
  cout << "Camera Number? (0 or 1) = ";
  cin >> cam_num;
  cout << endl;
  
  FlyCap2CVWrapper cam(cam_num);
  Mat frame;
  vector<vector<Point2f>> corner_list;

  //float PS=0.0053;
  //float F=8.0;
  
  Mat K=Mat::eye(3, 3, CV_32FC1);
  Mat D;
  
  K.at<float>(0, 0)=F/PS;
  K.at<float>(1, 1)=F/PS;
  K.at<float>(0, 2)=IW/2;
  K.at<float>(1, 2)=IH/2;

  cout << K << endl;
  cout << endl;

  while(true)
    {
      vector<Point2f> corner;
      frame = cam.readImage();
      if(frame.empty())
	{
	  cout << "Can't get image from Camera." << endl;
	  break;
	}
    
      Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      bool found = cv::findChessboardCorners(gray, cv::Size(CW, CH), corner,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

      if(found)
	cv::cornerSubPix(gray, corner, cv::Size(11, 11), cv::Size(-1, -1),
	  cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
	  30, 0.1));

      Mat draw, draws;
      frame.copyTo(draw);
      cv::drawChessboardCorners(draw, cv::Size(CW, CH), cv::Mat(corner), found);

      //cv::imshow("image",frame);
      cv::putText(draw, "SN: "+cam.getCameraSN(), cv::Point(10, 50),
		  cv::FONT_HERSHEY_TRIPLEX, 1.2, cv::Scalar(200, 100, 0), 1, cv::LINE_AA);
      cv::resize(draw, draws, cv::Size(0, 0), 0.75, 0.75);
      cv::imshow("Chessboard Corner", draws);
    
      int key = cv::waitKey(1);
      if(key == 'q')
	break;
      if(key == 27)
	break;
      if(key == 's' && found)
	{
	  corner_list.push_back(corner);
	  cout << "corner saved. " << corner_list.size() << endl;
	}
    }
  
  std::vector<cv::Point3f> object;
  for(int j=0; j<CH; j++)
    {
      for(int i=0; i<CW; i++)
	{
	  object.push_back(cv::Point3f(i*SQ_SIZE, j*SQ_SIZE, 0.0f));
	}
    }

  std::vector<std::vector<cv::Point3f> > object_list;
  for(std::size_t i=0;i<corner_list.size();i++){
    object_list.push_back(object);
  }
  
  std::vector<cv::Mat> R,T;

  double RMS = cv::calibrateCamera(object_list,corner_list, cv::Size(1280, 960),
				   K, D, R, T, CALIB_USE_INTRINSIC_GUESS
				   | CALIB_FIX_ASPECT_RATIO
				   | CALIB_ZERO_TANGENT_DIST
				   );
  
  cout << endl;
  cout << "RMS " << RMS << endl << endl;
  cout << "K " << K << endl << endl;
  cout << "D " << D << endl;
  
  cv::FileStorage fs("../data/"+cam.getCameraSN()+"_intrinsics.yml", cv::FileStorage::WRITE);
  fs << "SN" << cam.getCameraSN();
  fs << "RMS" << RMS;
  fs<<"K"<<K;
  fs<<"D"<<D;
  fs.release();

  destroyWindow("Chessboard Corner");
  
  return 0;
}


int main(int argc, char* argv[])
{
  char f;
  
  cout << endl;
  cout << "*** Camera Calibration ***" << endl;
  
  while(true)
    {
      cout << endl;
      cout << "Please Select" << endl;
      cout << " Check View (v)" << endl;
      cout << " Mono Calibration (m)" << endl;
      cout << " Stereo Calibration (s)" << endl;
      cout << " Check Stereo (c)" << endl;
      cout << " Check Mono Camera (n)" << endl;
      cout << " Quit (q)" << endl;
      cout << "Which ? = ";
      cin >> f;
      cout << endl;

      if(f=='v')
	check_view(argc,argv);

      else if(f=='s')
	stereo_calibrate(argc,argv);

      else if(f=='m')
	mono_calibrate(argc,argv);

      else if(f=='c')
	check_stereo(argc,argv);

      else if(f=='n')
	check_mono(argc,argv);

      else if(f=='q')
	break;
    }
  
  return 0;
}

