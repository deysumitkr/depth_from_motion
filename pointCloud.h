#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d.hpp"
//#include "five-point.cpp"
#include <iostream>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>

using namespace cv;

extern cv::Mat K, D;

int plotCloud(float p3D[], float c3D[], int len);

// container.cpp
void filterPoints(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, cv::Mat mask, std::vector<cv::Point2f> &newPoints1, std::vector<cv::Point2f> &newPoints2);
void showFlow(cv::Mat im, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t );
void drawEpipolarLines(cv::Mat im1, cv::Mat im2, cv::Mat F);
std::vector< cv::DMatch > findMatchingPoints(cv::Mat im1, cv::Mat im2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2);

int recoverPose( InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                     OutputArray _R, OutputArray _t, InputOutputArray _mask);



#endif
