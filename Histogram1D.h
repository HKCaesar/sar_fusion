//
// Created by aurora on 16-6-10.
//

#ifndef SAR_FUSION_HISTOGRAM1D_H
#define SAR_FUSION_HISTOGRAM1D_H

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

class Histogram1D {
public:
    Histogram1D();
    cv::MatND getHistogram(const cv::Mat &image);
    cv::Mat getHistogramImage(const cv::Mat &image);
private:
    int histSize[1];
    float hranges[2];
    const float* ranges[1];
    int channels[1];
};


#endif //SAR_FUSION_HISTOGRAM1D_H
