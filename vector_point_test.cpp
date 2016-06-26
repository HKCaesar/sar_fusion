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

void show_vector(vector<cv::Point> values){
    for(vector<cv::Point>::iterator iElement = values.begin(); iElement!=values.end(); iElement++){
        cout<<"x value is ="<<iElement->x<<" ,y value is = "<<iElement->y<< endl;
    }
    cout<<endl;
}

bool find_vec(vector<cv::Point> values, cv::Point val){
    bool flag = false;
    for(auto iElement=values.begin(); iElement!=values.end(); iElement++){
        if((val.x==iElement->x)&(val.y==iElement->y)){
            flag = true;
            return flag;
        }
    }
    return flag;
}


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

void init_mat(cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            datas[j] = 0;
        }
    }
}

void display_boundaries(vector<cv::Point>& boundary, int rows, int cols){
    cv::Mat mat_val(rows, cols, CV_8U);
    init_mat(mat_val);
    for(auto iElement=boundary.begin(); iElement!= boundary.end(); iElement++){
        mat_val.at<uchar>(iElement->y, iElement->x) = 1;
    }
    display_img(mat_val);
}

int main(){
    cv::Mat testapp(16, 16, CV_8U);
    vector<cv::Point> vec1, vec2;
    for(int i=0; i < testapp.rows; i++){
        for(int j=0; j< testapp.cols; j++){
            if((i<3)){
                if(i==0|j==0){
                    testapp.at<uchar>(i, j) = 3;
                }else{
                    if((i>=1&i<=3)&(j>=1&j<=2)){
                        testapp.at<uchar>(i, j) = 3;
                    }else{
                        testapp.at<uchar>(i, j) = 1;
                        vec1.push_back(cv::Point(j, i));
                    }
                }
            }else if((i>=3)&(j>4)){
                    testapp.at<uchar>(i, j) = 2;
                    vec2.push_back(cv::Point(j, i));
            }else{
                if(i==0|j==0){
                    testapp.at<uchar>(i, j) = 3;
                }else{
                    if((i>=1&i<=3)&(j>=1&j<=2)){
                        testapp.at<uchar>(i, j) = 3;
                    }else{
                        testapp.at<uchar>(i, j) = 1;
                        vec1.push_back(cv::Point(j, i));
                    }
                }
            }
        }
    }
    for(int i=5; i<=6; i++){
        for(int j=5; j<=7; j++){
            testapp.at<uchar>(i,j) = 1;
            vec1.push_back(cv::Point(j, i));
        }
    }

    for(int i=1; i<=3; i++){
        for(int j=1; j<=2; j++){
            testapp.at<uchar>(i,j) = 3;
//            vec1.push_back(cv::Point(j, i));
        }
    }

    display_img(testapp);
//    display_boundaries(vec1, 16, 16);
    vector<cv::Point> boundaries_add;
    vector<cv::Point> boundaries_sub;
    vector<cv::Point> boundaries;
    for(auto item = vec1.begin(); item!=vec1.end(); item++){

        bool temp_flag = true;
        bool temp_flag2 = true;
        cv::Point temp;
        cv::Point temp2;
        temp.x = item->x+1;
        temp.y = item->y+1;

        int boundary_x = item->x-1;
        int boundary_y = item->y-1;
        if(boundary_x>=0&boundary_y>=0){
            temp2.x = item->x-1;
            temp2.y = item->y-1;
            temp_flag2 = find_vec(vec1, temp2);
        }
        temp_flag = find_vec(vec1, temp);
        if(!temp_flag){
            boundaries_add.push_back(*item);
            boundaries.push_back(*item);
        }
        if(!temp_flag2){
            boundaries_sub.push_back(*item);
            boundaries.push_back(*item);
        }

    }

//    show_vector(boundaries_sub);

//    display_img(testapp);
    cout<< "________________________"<<endl;
    display_boundaries(boundaries_sub, 16, 16);
    return 0;
}