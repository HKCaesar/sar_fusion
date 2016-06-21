//
// Created by auroua on 16-6-16.
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
    // range from 3 to 15
    const int max_nij = 7;
    const int min_nij = 1;
    int init_nij = 4;
    int temp_nij = 0;
    const double sigma_n = 0.5227;

    string input_url = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg";
    string output_url = "/home/auroua/workspace/output.png";
    cv::Mat img_input = cv::imread(input_url);

    int type = img_input.type();
    cv::Size size = img_input.size();
    cv::Mat img_output(size, type);
    // store the n value of input pixel, the scale information
    cv::Mat img_n_val(size, CV_8U);
    int height = img_input.rows;
    int width = img_input.cols;

//    cout<<img_input.type()<<" tttttt "<<img_input.dims<<"    "<<(unsigned int)img_input.at<uchar>(0,1)<<"                "<<endl;
    cout<< img_output.type()<<endl;

    display_img(img_input);
    cout<<"_________________________________________"<<endl;

    int x_left_top = -1;  //left point x coordinate   rows
    int y_left_top = -1;  //left point y coordinate   cols
    int x_right_bottom = -1;   //x coordinate length
    int y_right_bottom = -1;   //y coordinate length
    int win_size = -1;  //windows size
    double cvij = -1;
    double threhold = 0;

    cv::Mat mean_output, std_output;

    // iterator the input image and generator the scale value of each pixel
    cv::Mat imageROI;
    for(int i=1; i<height-1; i++){
        uchar* data = img_input.ptr<uchar>(i);
        for(int j=1; j<width-1; j++){
            cout<<"index is === row="<<i<<" col="<<j<<" and the image values is "<<(unsigned int)data[j]<<endl;
            bool flag = true;
            init_nij = 4;
//            while(flag){
//                temp_nij = init_nij;
//                x_left_top = i - init_nij;
//                y_left_top = j - init_nij;
//                if(x_left_top<0 || y_left_top<0){
//                    init_nij = init_nij - 1;
//                    if(init_nij < min_nij){
//                        break;
//                    }else{
//                        continue;
//                    }
//                }
//                win_size = 2*init_nij + 1;
//                x_right_bottom = x_left_top + win_size;
//                y_right_bottom = y_left_top + win_size;
//                if(x_right_bottom > height or y_right_bottom > width){
//                    break;
//                }
//                cv::Mat imageROI(win_size, win_size, CV_8U);
////              imageROI = img_input(cv::Rect(y_left_top, x_left_top,  win_size, win_size));
//
//                for(int h=x_left_top; h<x_right_bottom; h++){
//                    for(int w=y_left_top; w<y_right_bottom; w++){
//                        imageROI.at<uchar>(h - x_left_top, w - y_left_top) = img_input.at<uchar>(h, w);
//                    }
//                }
//                cout <<imageROI.type()<< "    "<< imageROI.rows << " @@@@@@@@@@@@@@@@@@@@@@ "  << imageROI.cols<< endl;
//                display_img(imageROI);
////                cout<< "the mean val is ="<< cv::mean(imageROI)<<endl;
//                cv::meanStdDev(imageROI, mean_output, std_output);
//                cvij = std_output.at<double>(0,0)/mean_output.at<double>(0, 0);
//                threhold = sigma_n + 3*sqrt((1+2*sigma_n*sigma_n)/(2*win_size*win_size))*sigma_n;
//                cout << "the values is="<< mean_output.at<double>(0, 0) << " and the std output is ==="<< std_output.at<double>(0,0) << endl;
//                cout << "the sigma_cvij is=== "<<cvij<<" and the threahold is ==="<<threhold<< endl;
//                if(cvij>threhold){
//                    break;
//                }
//            }
            for(int n_val = min_nij; n_val <= max_nij; n_val++){
                x_left_top = i - n_val;
                y_left_top = j - n_val;
                if(x_left_top<0 || y_left_top<0){
                    break;
                }
                win_size = 2*n_val + 1;
                x_right_bottom = x_left_top + win_size;
                y_right_bottom = y_left_top + win_size;
                if(x_right_bottom > height or y_right_bottom > width){
                    break;
                }
                cv::Mat imageROI(win_size, win_size, CV_8U);
//              imageROI = img_input(cv::Rect(y_left_top, x_left_top,  win_size, win_size));

                for(int h=x_left_top; h<x_right_bottom; h++){
                    for(int w=y_left_top; w<y_right_bottom; w++){
                        imageROI.at<uchar>(h - x_left_top, w - y_left_top) = img_input.at<uchar>(h, w);
                    }
                }
                cout <<imageROI.type()<< "    "<< imageROI.rows << " @@@@@@@@@@@@@@@@@@@@@@ "  << imageROI.cols<< endl;
                display_img(imageROI);
//                cout<< "the mean val is ="<< cv::mean(imageROI)<<endl;
                cv::meanStdDev(imageROI, mean_output, std_output);
                cvij = std_output.at<double>(0,0)/mean_output.at<double>(0, 0);
                threhold = sigma_n + 3*sqrt((1+2*sigma_n*sigma_n)/(2*win_size*win_size))*sigma_n;
                cout << "the values is="<< mean_output.at<double>(0, 0) << " and the std output is ==="<< std_output.at<double>(0,0) << endl;
                cout << "the sigma_cvij is=== "<<cvij<<" and the threahold is ==="<<threhold<< endl;
                if(cvij>threhold){
                    break;
                }
            }
        }
    }







//    cv::namedWindow("sar");
//    cv::imshow("sar",img_input);
//    cv::waitKey(0);
    return 0;
}