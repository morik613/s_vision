#pragma once

#include <sstream>
#include "opencv2/opencv.hpp"
#include "flycapture/FlyCapture2.h"
#include "flycapture/FlyCapture2GUI.h"

class FlyCap2CVWrapper
{
protected:
    FlyCapture2::Camera flycam;
    FlyCapture2::CameraInfo flycamInfo;
    FlyCapture2::Error flycamError;
    FlyCapture2::Image flyImg, bgrImg;
    cv::Mat cvImg;
    FlyCapture2::BusManager busMgr;

public:
    FlyCap2CVWrapper(int number);
    ~FlyCap2CVWrapper();
    cv::Mat readImage();
    // Settings
    void autoExposure(bool flag, float absValue);
    void autoWhiteBalance(bool flag, int red, int blue);
    void autoSaturation(bool flag, float absValue);
    void autoShutter(bool flag, float ms);
    void autoGain(bool flag, float dB);
    void autoFrameRate(bool flag, float fps);
    bool checkError();
    std::string getCameraSN();
};
