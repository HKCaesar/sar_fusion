//
// Created by auroua on 16-6-21.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <string.h>

using namespace std;

void display_img(const cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        const uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            cout<<(unsigned int)datas[j]<<", ";
        }
        cout<<endl;
    }
}


int main(){
    string input_url = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg";
    string output_url = "/home/auroua/workspace/output16_16.png";
    cv::Mat img_input = cv::imread(input_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat img_output(16, 16, img_input.type());
    cv::resize(img_input, img_output, img_output.size());
    cv::imwrite(output_url, img_output);

    display_img(img_output);

    cv::namedWindow("sar");
    cv::imshow("sar",img_output);
    cv::waitKey(0);
}