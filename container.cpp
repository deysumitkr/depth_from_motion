#include "pointCloud.h"

cv::Mat K = (cv::Mat_<double>(3,3) << 	533.8946965168722, 0, 297.6972878158592, 
										0, 538.3821700365565, 241.3957474719554, 
										0, 0, 1);
										
cv::Mat D = (cv::Mat_<double>(5,1) <<	-0.09274903090291986, 0.09847045994005939, 0.002208733154636638, -0.006063225905350455, 0);

void filterPoints(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, cv::Mat mask, std::vector<cv::Point2f> &newPoints1, std::vector<cv::Point2f> &newPoints2){
	for(int i=0; i<mask.rows; i++){
		if ((int)mask.at<uchar>(i)){
			newPoints1.push_back(points1[i]);
			newPoints2.push_back(points2[i]);
		}
	}
}

void showFlow(cv::Mat im, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2){
	std::cout << points1.size() << std::endl;
	for (int i=0; i<points1.size(); i++){
		cv::circle(im, points1.at(i), 3, cv::Scalar(0,0,220), 1);
		cv::line(im, points1.at(i), points2.at(i), cv::Scalar(0, 220, 200), 1);
	}
	cv::imshow("Optical Flow", im);
	cv::waitKey(0);
	cv::destroyWindow("Optical Flow");
}

void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
{
    Mat E = _E.getMat().reshape(1, 3);
    CV_Assert(E.cols == 3 && E.rows == 3);

    Mat D, U, Vt;
    SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    Mat R1, R2, t;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;

    R1.copyTo(_R1);
    R2.copyTo(_R2);
    t.copyTo(_t);
}


void drawEpipolarLines(cv::Mat im1, cv::Mat im2, cv::Mat F){
	std::vector<cv::Point2f> points1, points2;
	for (int i=0; i<im1.rows; i=i+40 ){
		points1.push_back(cv::Point2f(300, i));
	}
	//std::cout << points1 << std::endl;
	cv::Mat epiLines1, epiLines2;
	cv::computeCorrespondEpilines(points1, 1, F, epiLines2);
	//std::cout << epiLines2 << epiLines2.rows <<  std::endl;
	//std::cout << epiLines2.at<float>(0,0) << std::endl;
	
	for (int i=0; i<epiLines2.rows; i++){
		float x1=0.0, x2=im2.cols-1;
		float y1 = -1.0*(epiLines2.at<float>(i,0)*x1 + epiLines2.at<float>(i,2))/epiLines2.at<float>(i,1);
		float y2 = -1.0*(epiLines2.at<float>(i,0)*x2 + epiLines2.at<float>(i,2))/epiLines2.at<float>(i,1);
		cv::line(im2, cv::Point2f(x1,y1), cv::Point2f(x2,y2), cv::Scalar::all(-1), 1, CV_AA);
		points2.push_back(cv::Point2f(x1, y1));
	}
	
	cv::computeCorrespondEpilines(points2, 2, F, epiLines1);
	for (int i=0; i<epiLines1.rows; i++){
		float x1=0.0, x2=im1.cols-1;
		float y1 = -1.0*(epiLines1.at<float>(i,0)*x1 + epiLines1.at<float>(i,2))/epiLines1.at<float>(i,1);
		float y2 = -1.0*(epiLines1.at<float>(i,0)*x2 + epiLines1.at<float>(i,2))/epiLines1.at<float>(i,1);
		cv::line(im1, cv::Point2f(x1,y1), cv::Point2f(x2,y2), cv::Scalar::all(-1), 1, CV_AA);
	}
	
	cv::imshow("Epilines Img1", im1);
	cv::imshow("Epilines Img2", im2);
	cv::waitKey(0);
	cv::destroyWindow("Epilines Img1");
	cv::destroyWindow("Epilines Img2");
}

std::vector< cv::DMatch > findMatchingPoints(cv::Mat im1, cv::Mat im2, 
												std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2){
	// convert to gray and show
	cv::Mat im_gray1, im_gray2;
	cv::cvtColor(im1, im_gray1, CV_BGR2GRAY);
	cv::cvtColor(im2, im_gray2, CV_BGR2GRAY);
	
	
	//-- Step 1: Surf detector (find keypoints)
	cv::SurfFeatureDetector detector( (int)100 );
	//std::vector<cv::KeyPoint> keypoints1, keypoints2;

	detector.detect( im_gray1, keypoints1 );
	detector.detect( im_gray2, keypoints2 );
	
	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SurfDescriptorExtractor extractor;

	cv::Mat descriptors1, descriptors2;

	extractor.compute( im_gray1, keypoints1, descriptors1 );
	extractor.compute( im_gray2, keypoints2, descriptors2 );
	
	/*
	//-- Draw keypoints
	cv::Mat img_keypoints_1, img_keypoints_2;

	cv::drawKeypoints( im1, keypoints1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::drawKeypoints( im2, keypoints2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

	//-- Show detected (drawn) keypoints
	cv::imshow("Keypoints 1", img_keypoints_1 );
	cv::imshow("Keypoints 2", img_keypoints_2 );

	cv::waitKey(0);
	*/
	
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );
	
	return matches;
}

int recoverPose( InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                     OutputArray _R, OutputArray _t, InputOutputArray _mask)
{
    Mat points1, points2, cameraMatrix;
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0,0);
    double fy = cameraMatrix.at<double>(1,1);
    double cx = cameraMatrix.at<double>(0,2);
    double cy = cameraMatrix.at<double>(1,2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    Mat Q;
    triangulatePoints(P0, P1, points1, points2, Q);
    Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    triangulatePoints(P0, P2, points1, points2, Q);
    Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    triangulatePoints(P0, P3, points1, points2, Q);
    Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    triangulatePoints(P0, P4, points1, points2, Q);
    Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty())
    {
        Mat mask = _mask.getMat();
        CV_Assert(mask.size() == mask1.size());
        bitwise_and(mask, mask1, mask1);
        bitwise_and(mask, mask2, mask2);
        bitwise_and(mask, mask3, mask3);
        bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask4.copyTo(_mask);
        return good4;
    }
}
