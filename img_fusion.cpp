//
// Created by auroua on 16-6-26.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <string.h>
#include <map>
#include <list>
#include <set>
#include <math.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>


using namespace std;

void display_img(const cv::Mat& image){
    for(int i = 0; i < image.rows; i++){
        const uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            cout<<(unsigned int)datas[j]<<", ";
        }
        cout<<endl;
    }
}

int find_top(cv::Mat& img1){
    int vals;
    for(int i=0; i< img1.rows; i++){
        const uchar* datas = img1.ptr<uchar>(i);
        for(int j=0; j<img1.cols; j++){
            vals = (unsigned int)datas[j];
            if(vals!=0){
                return i;
            }
        }
    }
    return 0;
}

double cacle_mean(cv::Mat& img){
    double sum = 0;
    double total = img.rows*img.cols;
    for(int i = 0; i < img.rows; i++){
        const uchar* datas = img.ptr<uchar>(i);
        for(int j = 0; j < img.cols; j++){
            sum += (unsigned int)datas[j];
        }
    }
    return  sum/total;
}

double cacle_target_mean(cv::Mat& img){
    double sum = 0;
    double total =0;
    for(int i = 0; i < img.rows; i++){
        const uchar* datas = img.ptr<uchar>(i);
        for(int j = 0; j < img.cols; j++){
            if((unsigned int)datas[j]!=0){
                sum += (unsigned int)datas[j];
                total++;
            }
        }
    }
    return  sum/total;
}


int find_bottom(cv::Mat& img1){
    int vals;
    for(int i=img1.rows-1; i>0; i--){
        const uchar* datas = img1.ptr<uchar>(i);
        for(int j=0; j<img1.cols; j++){
            vals = (unsigned int)datas[j];
            if(vals!=0){
                return i;
            }
        }
    }
    return 0;
}

int find_left(cv::Mat& img1){
    int vals;
    int left = img1.cols;
    for(int i=0; i<img1.rows; i++){
        const uchar* datas = img1.ptr<uchar>(i);
        for(int j=0; j<img1.cols; j++){
            vals = (unsigned int)datas[j];
            if(vals!=0){
                if(left>j){
                    left = j;
                }
            }
        }
    }
    return left;
}


int find_right(cv::Mat& img1){
    int vals;
    int right = 0;
    for(int i=0; i<img1.rows; i++){
        const uchar* datas = img1.ptr<uchar>(i);
        for(int j=0; j<img1.cols; j++){
            vals = (unsigned int)datas[j];
            if(vals!=0){
                if(right<j){
                    right = j;
                }
            }
        }
    }
    return right;
}

int main(){
//    cv::Mat test_mat(10, 10, CV_8U);
//    for(int i=0; i<10; i++){
//        for(int j=0; j<10; j++){
//            test_mat.at<uchar>(i, j) = 1;
//        }
//    }
//    display_img(test_mat);
//    cout<<"the mean is==="<<cacle_mean(test_mat);
//    test_mat = test_mat*2;
//    display_img(test_mat);

    string input_url = "/home/auroua/Desktop/samples/bmp2/HB03352_000b.png";
    string input_url2 = "/home/auroua/Desktop/samples/2S1/HB19873b.png";
    string bg_url = "/home/auroua/Desktop/samples/hb06232.jpg";
    cv::Mat img_input1 = cv::imread(input_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat img_input2 = cv::imread(input_url2, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat bg_input = cv::imread(bg_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


    double bg_mean = cacle_mean(bg_input);
    double img1_mean = cacle_target_mean(img_input1);
    double img2_mean = cacle_target_mean(img_input2);

//    img_input1 = img_input1+bg_mean;
    cout<<"bg mean ="<<bg_mean<< " , img1_mean="<< img1_mean<<", img2_mean=="<<img2_mean<< "the divide is"<<bg_mean/img1_mean<<endl;
    img_input1 = img_input1*(bg_mean/img1_mean);
    img_input1 = img_input1+60;

    img_input2 = img_input2*(bg_mean/img2_mean);
//    img_input2 = img_input2+60;
//    display_img(img_input1);
    cout<<"new img1_mean is ==="<< cacle_mean(img_input1)<<endl;
//    cv::imwrite(output_url, visual_mat);
//    display_img(img_input1);
    int left_r1 = find_top(img_input1);
    int left_c1 = find_left(img_input1);
    int right_r1 = find_bottom(img_input1);
    int right_c1 = find_right(img_input1);
    cout<< "==="<<left_r1<<" , "<<left_c1<<"  , "<<right_r1<<" , "<<right_c1<<endl;

    int left_r2 = find_top(img_input2);
    int left_c2 = find_left(img_input2);
    int right_r2 = find_bottom(img_input2);
    int right_c2 = find_right(img_input2);
    cout<< "==="<<left_r2<<" , "<<left_c2<<"  , "<<right_r2<<" , "<<right_c2<<endl;


//    for(int y=139; y<bg_input.rows; y++){
//        for(int x=1173; x<bg_input.cols; x++){
    int y_bg = 139;
    int x_bg = 1173;
    for(int img_y1 = left_r1; img_y1<=right_r1; img_y1++, y_bg++){
        x_bg = 1173;
        for(int img_x1 = left_c1; img_x1<=right_c1; img_x1++, x_bg++){
            if((unsigned int)img_input1.at<uchar>(img_y1, img_x1)!=60){
                bg_input.at<uchar>(y_bg, x_bg) = img_input1.at<uchar>(img_y1, img_x1);
            }
        }
    }

    y_bg = 1390;
    x_bg = 1237;
    for(int img_y1 = left_r2; img_y1<=right_r2; img_y1++, y_bg++){
        x_bg = 1237;
        for(int img_x1 = left_c2; img_x1<=right_c2; img_x1++, x_bg++){
            if((unsigned int)img_input1.at<uchar>(img_y1, img_x1)!=60){
                bg_input.at<uchar>(y_bg, x_bg) = img_input1.at<uchar>(img_y1, img_x1);
            }
        }
    }

    cv::imwrite("/home/auroua/bg_img.jpg", bg_input);
    cv::imshow( "bg img", bg_input );
    cv::waitKey( 0 );

    return 0;
}