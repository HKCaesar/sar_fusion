#include <QApplication>

#include "mainwindow.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
//    http://stackoverflow.com/questions/27533203/how-do-i-use-sift-in-opencv-3-0-with-c

//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();
//
//    return a.exec();
    Mat img_1 = imread("/home/auroua/workspace/lena.jpeg");
//    cv::drawKeypoints (image,keypoints,image,cv::Scalar::all (255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // now, you can no more create an instance on the 'stack', like in the tutorial
    // (yea, noticed for a fix/pr).
    // you will have to use cv::Ptr all the way down:
    //
    Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //cv::Ptr<Feature2D> f2d = ORB::create();
    // you get the picture, i hope..

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect( img_1, keypoints_1 );
//    f2d->detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
//    f2d->compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector<DMatch> matches;
//    matcher.match( descriptors_1, descriptors_2, matches );

    cv::drawKeypoints(img_1,keypoints_1,img_1,cv::Scalar::all(255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("surf");
    cv::imshow("surf",img_1);
    cv::waitKey(0);
    return 0;
}
