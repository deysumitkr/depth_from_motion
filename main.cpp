#include "pointCloud.h"

using namespace std;

cv::Mat _P3D_;

void drawAR(void* param){
	glBegin(GL_POINTS);
    glColor3f(0.1, 0.8, 0.3);
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glVertex3f(0, 1, 0);
    
    for (int i=0; i<_P3D_.rows; i++){
    	glVertex3f(_P3D_.at<float>(i,0), 
    				_P3D_.at<float>(i,1),
    				_P3D_.at <float>(i,2));
    }
    
    glEnd();
    glPointSize(2.0);
}

void glPractice(){
	std::string win = "GL Practice";
	cv::namedWindow(win, cv::WINDOW_OPENGL);
	cv::setOpenGlContext(win);
	cv::setOpenGlDrawCallback(win, drawAR, NULL);
	cv::updateWindow(win);
	cv::waitKey(0);
}


int main(int argc, char** argv){
	// Load and show two images
	std::string path1 = "../images/chawanL.png";
	std::string path2 = "../images/chawanR.png";
	
	cv::Mat dim1 = cv::imread(path1);
	cv::Mat dim2 = cv::imread(path2);

	// Undistort images
	cv::Mat im1, im2;
	cv::undistort(dim1, im1, K, D);
	cv::undistort(dim2, im2, K, D);

	// Find matching points in two images
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector <cv::DMatch> matches = findMatchingPoints(im1, im2, keypoints1, keypoints2);
		
	std::vector<cv::Point2f> match1;
	std::vector<cv::Point2f> match2;

	for( int i = 0; i < matches.size(); i++ ){
		match1.push_back( keypoints1[ matches[i].queryIdx ].pt );
		match2.push_back( keypoints2[ matches[i].trainIdx ].pt );
	}

	// Find Fundamental Matrix - RANSAC
	cv::Mat mask;
	cv::Mat F = cv::findFundamentalMat(match1, match2, cv::FM_RANSAC, 3., 0.99, mask);
	std::cout << "\nF (RANSAC) = \n" << F << std::endl;
	
	std::vector<cv::Point2f> goodPoints1, goodPoints2;
	filterPoints(match1, match2, mask, goodPoints1, goodPoints2);
	
	// Refine Fundamental Matrix - 8point Algorithm using refined points
	F = cv::findFundamentalMat(goodPoints1, goodPoints2, cv::FM_8POINT);
	std::cout << "\nF (8-Point) = \n" << F << std::endl;
	
	//showFlow(im1.clone(), goodPoints1, goodPoints2);

	if(false){
		cv::Mat all_matches;
		cv::Mat im_matches;
		cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, im_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, all_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::imshow("Good Matches", im_matches);
		cv::imshow("All Matches", all_matches);
		cv::waitKey(0);
	}

	drawEpipolarLines(im1, im2, F);
	
	cv::Mat E = K.t() * F * K;
	std::cout << "\nEssential Matrix = \n" << E << std::endl;
	
	cv::Mat Rc, Tc, maskEP;
	recoverPose(E, goodPoints1, goodPoints2, K, Rc, Tc, maskEP);
	std::cout << "\nCamera Rotation = \n" << Rc << std::endl;
	std::cout << "\nCamera Translation = \n" << Tc << std::endl;
	
	cv::Mat R1, R2, T;
	decomposeEssentialMat(E, R1, R2, T);
	
	std::cout << "\nRotation & Translation Matrices = \n";
	std::cout << R1 << std::endl;
	std::cout << R2 << std::endl;
	std::cout << T  << std::endl;

	// new optimal camera matrix not used
	cv::Mat newK = cv::getOptimalNewCameraMatrix(K, D, im1.size(), 0.0);
	std::cout << "\nCamera Matrix = \n" << K << std::endl;
	std::cout << "\nNew Camera Matrix = \n" << newK << std::endl;
	
	// Uncalibrated Stereo Rectification
	// http://ece631web.groups.et.byu.net/Lectures/ECEn631%2014%20-%20Calibration%20and%20Rectification.pdf
	cv::Mat H1, H2;
	cv::stereoRectifyUncalibrated(goodPoints1, goodPoints2, F, im1.size(), H1, H2);
	
	cv::Mat stereoR1, stereoR2;
	stereoR1 = K.inv()*H1*K; 
	stereoR2 = K.inv()*H2*K;
	
	cv::Mat cam1Map1, cam1Map2, cam2Map1, cam2Map2;
	cv::initUndistortRectifyMap(K, D, stereoR1, K, im1.size(), CV_32FC1, cam1Map1, cam1Map2 );
	cv::initUndistortRectifyMap(K, D, stereoR2, K, im2.size(), CV_32FC1, cam2Map1, cam2Map2 );
	
	cv::Mat rectifiedIm1, rectifiedIm2;
	cv::remap(im1, rectifiedIm1, cam1Map1, cam1Map2, CV_INTER_LINEAR);
	cv::remap(im2, rectifiedIm2, cam2Map1, cam2Map2, CV_INTER_LINEAR);
	
	cv::Mat rectified;
	cv::hconcat(rectifiedIm1, rectifiedIm2, rectified);
	cv::imshow("Rectified", rectified);
	
	// finding Q matrix
	//cv::Mat g1, g2, g3, g4, Q;
	//cv::stereoRectify(K, D, K, D, im1.size(), Rc, Tc, g1, g2, g3, g4, Q);
	
	//--Disparity
	cv::StereoBM SBM(cv::StereoBM::BASIC_PRESET, 16*1, 19);
	/*
	SBM.state->SADWindowSize = 15;
    SBM.state->numberOfDisparities = 160;
    SBM.state->preFilterSize = 5;
    SBM.state->preFilterCap = 1;
    SBM.state->minDisparity = 0;
    SBM.state->textureThreshold = 5;
    SBM.state->uniquenessRatio = 5;
    SBM.state->speckleWindowSize = 0;
    SBM.state->speckleRange = 20;
    SBM.state->disp12MaxDiff = 64;
	*/
	cv::Mat gray1, gray2, disparity, disparityImg;
	cv::cvtColor(rectifiedIm1, gray1, CV_BGR2GRAY);
	cv::cvtColor(rectifiedIm2, gray2, CV_BGR2GRAY);
	SBM(gray1, gray2, disparity);
	
	double minVal, maxVal;
	minMaxLoc( disparity, &minVal, &maxVal );
	printf("\nMin disp: %f Max value: %f \n", minVal, maxVal);
	disparity.convertTo( disparityImg, CV_8UC1, 255/(maxVal - minVal));
	
	
	cv::imshow("Disparity", disparityImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
#if 0	
	cv::Mat depth;
	cv::reprojectImageTo3D(disparityImg, depth, Q);
	
	std::cout << "\ndepth size = " << depth.size() << std::endl;
	std::cout << "\ndepth data =\n" << depth.at<float>(255, 430) << std::endl;	
	
	
#endif
	cv::Mat P0 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
									 		0, 1, 0, 0,
									 		0, 0, 1, 0);
	
	cv::Mat P1, P2;
	cv::hconcat(R1, T, P1);
	cv::hconcat(R2, T, P2);
	
	cv::Mat M0 = K*P0;
	cv::Mat M1 = K*P1;
	cv::Mat M2 = K*P2;
	
	std::cout << "M done!" << std::endl;
	cv::Mat points3D, points3DH;
	cv::triangulatePoints(M0, M2, goodPoints1, goodPoints2, points3DH);
	cv::convertPointsFromHomogeneous(points3DH.t(), points3D);
	
	std::cout << "inhomgeneous done!" << std::endl;
	points3D = 1.0*points3D; // scaling
	//float* points3Df = (float*)points3D.data; //convert mat to float pointer array
	
	//_P3D_ = points3D.clone();
	
	//glPractice();
	
	std::vector<float> vertices, colors;
	for (int i=0; i<points3D.rows; i++){
		for (int j=0; j<3; j++){
			vertices.push_back( points3D.at<float>(i,j) );
			colors.push_back((j==1)?1.0:0.0); // all green
			
			// RGB - Color
			/*
			colors.push_back( (float)((float)im1.at<cv::Vec3b>(goodPoints1.at(i).y, goodPoints1.at(i).x)[0]/255.0) ); //Blue
			colors.push_back( (float)((float)im1.at<cv::Vec3b>(goodPoints1.at(i).y, goodPoints1.at(i).x)[1]/255.0) ); //Green
			colors.push_back( (float)((float)im1.at<cv::Vec3b>(goodPoints1.at(i).y, goodPoints1.at(i).x)[2]/255.0) ); //Red
			*/
		}
	}
	
	int numVertices = points3D.rows;


	std::cout << "Number of vertices: " << vertices.size()/3.0 << std::endl;
	std::cout << "Number of color vecs: " << colors.size()/3.0 << std::endl;
	
	float* verticesf = &vertices[0];
	float* colorsf = &colors[0];

	//std::cout << "\n3D Coordinates = \n" << points3D(cv::Rect(0,0,1,10)) << std::endl;
	std::cout << verticesf[0] << verticesf[1] << verticesf[2]  << std::endl;
	std::cout << colorsf[0] << colorsf[1] << colorsf[2]  << std::endl;
	plotCloud(verticesf, colorsf, numVertices);
	
	
	return 0;
}
