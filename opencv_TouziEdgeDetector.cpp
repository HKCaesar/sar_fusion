//
// Created by auroua on 16-6-16.
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <string.h>

using namespace std;

const double CONST_PI =       3.14159265358979323846;  /* pi */
const double CONST_PI_2 =     1.57079632679489661923;  /* pi/2 */
const double CONST_PI_4 =     0.78539816339744830962;  /* pi/4 */
const double CONST_PI_8 =     0.39269908169872415481;  /* pi/8 */
const double THRESHOLD = 0.3;


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

void display_img2(const cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        const uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            cout<<(unsigned int)datas[j]<<endl;
        }
//        cout<<endl;
    }
}

void display_img_double(const cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        const double* datas = image.ptr<double>(i);
        for(int j = 0; j < image.cols; j++){
            cout<<(double)datas[j]<<", ";
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

void caculate_touziEdge(const cv::Mat& image, const int radius){
    const int NB_DIR = 4;
    // Number of region of the filter
    const int NB_REGION = 2;
    // Definition of the 4 directions
    double Theta[NB_DIR];
    Theta[0] = 0.;
    Theta[1] = CONST_PI_4;
    Theta[2] = CONST_PI_2;
    Theta[3] = 3 * CONST_PI / 4.;
    // contains for the 4 directions the sum of the pixels belonging to each region
    double Sum[NB_DIR][NB_REGION];
    // Mean of region 1
    double M1;
    // Mean of region 2
    double M2;
    // Result of the filter for each direction
    double R_theta[NB_DIR];
    double Sum_R_theta = 0.;
    // Intensity of the contour
    double R_contour;
    // Direction of the contour
    double Dir_contour = 0.;
    // sign of the contour
    int sign;
    // Pixel location in the input image
    int x;
    int y;
    // Location of the central pixel in the input image
    int xc;
    int yc;
    int cpt = 0;

    cv::Mat img_touiz(image.size(), CV_64FC1);

    for(int i=1; i < image.rows-1; i++){
        for(int j=1; j< image.cols-1; j++){
            xc = i;
            yc = j;
            // Initializations
            for (int dir = 0; dir < NB_DIR; ++dir)
            {
                for (int m = 0; m < NB_REGION; m++)
                    Sum[dir][m] = 0.;
            }

            R_contour = 1;
            Dir_contour = 0.;
            Sum_R_theta = 0.;
            for(int irow=-1; irow<=1; irow++){
                for(int jcol=-1; jcol<=1; jcol++){
                    x = xc+irow;
                    y = yc+jcol;

                    // We determine for each direction with which region the pixel belongs.

                    // Horizontal direction
                    if (y < yc) Sum[0][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
                    else if (y > yc) Sum[0][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

                    // Diagonal direction 1
                    if ((y - yc) < (x - xc)) Sum[1][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
                    else if ((y - yc) > (x - xc)) Sum[1][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

                    // Vertical direction
                    if (x > xc) Sum[2][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
                    else if (x < xc) Sum[2][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

                    // Diagonal direction 2
                    if ((y - yc) > -(x - xc)) Sum[3][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
                    else if ((y - yc) < -(x - xc)) Sum[3][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));
                }
            }

            // Loop on the 4 directions
            for (int dir = 0; dir < NB_DIR; ++dir)
            {
                // Calculation of the mean of the 2 regions
                M1 = Sum[dir][0] / static_cast<double>(radius * (2 * radius + 1));
                M2 = Sum[dir][1] / static_cast<double>(radius * (2 * radius + 1));

                // Calculation of the intensity of the contour
//                double values = M1/M2;
//                std::cout<< "m1/m2 value is ===" << values<<endl;
                if ((M1 != 0) && (M2 != 0)) R_theta[dir] = static_cast<double>(std::min((M1 / M2), (M2 / M1)));
//                if ((M1 != 0) && (M2 != 0)) R_theta[dir] = static_cast<double>(std::min((M1 / M2), (M2 / M1)));
                else R_theta[dir] = 0.;

                // Determination of the maximum intensity of the contour
                R_contour = static_cast<double>(std::min(R_contour, R_theta[dir]));

                // Determination of the sign of contour
                if (M2 > M1) sign = +1;
                else sign = -1;

                Dir_contour += sign * Theta[dir] * R_theta[dir];
                Sum_R_theta += R_theta[dir];

            } // end of the loop on the directions

            // Assignment of this value to the output pixel
//            it.Set(static_cast<OutputPixelType>(R_contour));
            img_touiz.at<double>(i, j) = R_contour;
        }
    }
    display_img_double(img_touiz);
//    cv::namedWindow("sar");
//    cv::imshow("sar",img_touiz);
//    cv::waitKey(0);
}

double caculate_touziEdge_pixel(const cv::Mat& image, const int radius, const int xc, const int yc){
    const int NB_DIR = 4;
    // Number of region of the filter
    const int NB_REGION = 2;
    // Definition of the 4 directions
    double Theta[NB_DIR];
    Theta[0] = 0.;
    Theta[1] = CONST_PI_4;
    Theta[2] = CONST_PI_2;
    Theta[3] = 3 * CONST_PI / 4.;
    // contains for the 4 directions the sum of the pixels belonging to each region
    double Sum[NB_DIR][NB_REGION];
    // Mean of region 1
    double M1;
    // Mean of region 2
    double M2;
    // Result of the filter for each direction
    double R_theta[NB_DIR];
    double Sum_R_theta = 0.;
    // Intensity of the contour
    double R_contour;
    // Direction of the contour
    double Dir_contour = 0.;
    // sign of the contour
    int sign;
    // Pixel location in the input image
    int x;
    int y;
//    cv::Mat img_touiz(image.size(), CV_64FC1);

    // Initializations
    for (int dir = 0; dir < NB_DIR; ++dir)
    {
        for (int m = 0; m < NB_REGION; m++)
            Sum[dir][m] = 0.;
    }

    R_contour = 1;
    Dir_contour = 0.;
    Sum_R_theta = 0.;
    for(int irow=-1*radius; irow<=radius; irow++){
        for(int jcol=-1*radius; jcol<=radius; jcol++){
            x = xc+irow;
            y = yc+jcol;

            // We determine for each direction with which region the pixel belongs.
            // Horizontal direction
            if (y < yc) Sum[0][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
            else if (y > yc) Sum[0][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

            // Diagonal direction 1
            if ((y - yc) < (x - xc)) Sum[1][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
            else if ((y - yc) > (x - xc)) Sum[1][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

            // Vertical direction
            if (x > xc) Sum[2][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
            else if (x < xc) Sum[2][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));

            // Diagonal direction 2
            if ((y - yc) > -(x - xc)) Sum[3][0] += static_cast<double>((unsigned)image.at<uchar>(x, y));
            else if ((y - yc) < -(x - xc)) Sum[3][1] += static_cast<double>((unsigned)image.at<uchar>(x, y));
        }
    }

    // Loop on the 4 directions
    for (int dir = 0; dir < NB_DIR; ++dir)
    {
        // Calculation of the mean of the 2 regions
        M1 = Sum[dir][0] / static_cast<double>(radius * (2 * radius + 1));
        M2 = Sum[dir][1] / static_cast<double>(radius * (2 * radius + 1));

        // Calculation of the intensity of the contour
//      double values = M1/M2;
//      std::cout<< "m1/m2 value is ===" << values<<endl;
        if ((M1 != 0) && (M2 != 0)) R_theta[dir] = static_cast<double>(std::min((M1 / M2), (M2 / M1)));
//      if ((M1 != 0) && (M2 != 0)) R_theta[dir] = static_cast<double>(std::min((M1 / M2), (M2 / M1)));
        else R_theta[dir] = 0.;
        // Determination of the maximum intensity of the contour
        R_contour = static_cast<double>(std::min(R_contour, R_theta[dir]));

        // Determination of the sign of contour
        if (M2 > M1) sign = +1;
        else sign = -1;

        Dir_contour += sign * Theta[dir] * R_theta[dir];
        Sum_R_theta += R_theta[dir];

    } // end of the loop on the directions
    return R_contour;
//    display_img_double(img_touiz);
//    cv::namedWindow("sar");
//    cv::imshow("sar",img_touiz);
//    cv::waitKey(0);
}

bool contain_edge(const cv::Mat& img_edge, int x_left, int y_left, int height, int width){
    unsigned int value = 0;
    for(int i = x_left; i<x_left+height; i++){
        for(int j=y_left; j<y_left+width; j++){
            value = (unsigned int)img_edge.at<uchar>(i, j);
            if(value>0){
                return true;
            }
        }
    }
    return false;
}

void fillin(cv::Mat& img_edge, int x_left, int y_left, int height, int width, int values){
    for(int i = x_left; i<x_left+height; i++){
        for(int j=y_left; j<y_left+width; j++){
            img_edge.at<uchar>(i, j) = values;
        }
    }
}

void contain_one_count(const cv::Mat& img_edge){
    unsigned int value = 0;
    for(int i = 0; i<img_edge.rows; i++){
        for(int j=0; j<img_edge.cols; j++){
            if((unsigned int)img_edge.at<uchar>(i, j)==1){
                value++;
            }
        }
    }
    cout<<value<<endl;
}


void normalize(const cv::Mat& img_edge, cv::Mat& img_dst, int totalcount){
    for(int i = 0; i<img_edge.rows; i++){
        for(int j=0; j<img_edge.cols; j++){
            img_dst.at<double>(i,j) = (unsigned int)img_edge.at<uchar>(i, j)/ static_cast<double>(totalcount);
        }
    }
}

void init_mat_value(cv::Mat& image, int value){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            datas[j] = value;
        }
    }
}

void reverse_mat_value(cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    int temp = 0;
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            temp = (int)image.at<uchar>(i, j);
            if(temp!=0){
                image.at<uchar>(i, j) = 0;
            }else{
                image.at<uchar>(i, j) = 1;
            }
        }
    }
}

int main(){
    // range from 3 to 15
    const int max_nij = 7;
    const int min_nij = 1;
    const int init_nij = 4;
    int temp_nij = 0;
    int status_nij = 0;
    const double sigma_n = 0.5227;

    string input_url = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg";
//    string input_url = "/home/auroua/workspace/output16_16.png";
    string output_url = "/home/auroua/workspace/output.png";
    cv::Mat img_input = cv::imread(input_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    int type = img_input.type();
    cv::Size size = img_input.size();
    cv::Mat img_output(size, type);
    cv::Mat img_touiz(size, CV_64FC1);
    cv::Mat img_edge(size, type);
    cv::Mat img_region(size, CV_8S);
    init_mat(img_output);
    init_mat(img_touiz);
    init_mat(img_edge);
    init_mat(img_region);
    // store the n value of input pixel, the scale information
    cv::Mat img_n_val(size, CV_8U);
    int height = img_input.rows;
    int width = img_input.cols;

    cout<<img_input.type()<<" tttttt "<<img_input.dims<<"    "<<(unsigned int)img_input.at<uchar>(0,1)<<"                "<<endl;
//    cout<< img_output.type()<<endl;

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
    // Adaptive to fit the size
    cv::Mat imageROI;
    bool flag = true;
    bool flag_left = false;
    bool flag_right = false;
    bool flag_inc = false;
    bool flag_sub = false;
    for(int i=1; i<height-1; i++){
        uchar* data = img_input.ptr<uchar>(i);
        for(int j=1; j<width-1; j++){
            cout<<"index is === row="<<i<<" col="<<j<<" and the image values is "<<(unsigned int)data[j]<<endl;
            flag = true;
            flag_left = false;
            flag_right = false;
            flag_inc = false;
            flag_sub = false;
            status_nij = init_nij;
            while(flag){
                temp_nij = status_nij;
                x_left_top = i - status_nij;
                y_left_top = j - status_nij;
                if(x_left_top<0 || y_left_top<0){
                    status_nij = max(status_nij - 1, min_nij);
                    flag_left = true;
                    continue;
                }
                win_size = 2*status_nij + 1;
                x_right_bottom = x_left_top + win_size;
                y_right_bottom = y_left_top + win_size;
                if(x_right_bottom > height or y_right_bottom > width){
                    status_nij = max(status_nij - 1, min_nij);
                    flag_right = true;
                    continue;
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
                if(cvij<=threhold){
                    if(flag_left || flag_right){
                        break;
                    }else{
                        if(flag_sub){
                            break;
                        }else{
                            status_nij = status_nij+1;
                            flag_inc = true;
                            if(status_nij>max_nij){
                                break;
                            }
                        }
                    }
                }else{
                    if(flag_inc){
                        break;
                    }else{
                        status_nij = status_nij - 1;
                        flag_sub =  true;
                        if(status_nij < min_nij){
                            break;
                        }
                    }

                }
            }
            img_output.at<uchar>(i, j) = temp_nij;
        }
    }
    cout<<"------------------------------------------------------------"<<endl;
//    display_img2(img_input);

    // caculate obt touziedge
//    caculate_touziEdge(img_input, 1);
//    double values = caculate_touziEdge_pixel(img_input, 1, 1, 1);
//    cout<< values<< endl;
    unsigned int scale = 0;
    double result = 0;
    for(int i=1; i< img_output.rows-1; i++){
        for(int j=1; j< img_output.cols-1; j++){
            scale = (unsigned int)img_output.at<uchar>(i,j);
            result = caculate_touziEdge_pixel(img_input, scale, i, j);
//            cout<< result << " , "<<endl;
            img_touiz.at<double>(i,j) = result;
            if(result < THRESHOLD){
                img_edge.at<uchar>(i,j) = 1;
            }
        }
    }
    display_img_double(img_touiz);
    cout<<"----------------------------------------------------------------------"<<endl;
    display_img(img_edge);

    //image segmentation
    int segment_value = 0;
    bool flag_segment = false;
    for(int i=0; i< img_input.rows; i=i+16){
        for(int j=0;j<img_input.cols; j=j+16){
            flag_segment = false;
            flag_segment = contain_edge(img_edge, i, j, 16, 16);
            if(flag_segment){
                for(int i_8 = i; i_8 < i+16; i_8=i_8+8){
                    for(int j_8 = j; j_8 < j+16;  j_8 = j_8+8){
                        flag_segment = false;
                        flag_segment = contain_edge(img_edge, i_8, j_8, 8, 8);
                        if(flag_segment){
                            for(int i_4 = i_8; i_4 < i_8+8; i_4=i_4+4){
                                for(int j_4 = j_8; j_4 < j_8+8;  j_4 = j_4+4){
                                    flag_segment = false;
                                    flag_segment = contain_edge(img_edge, i_4, j_4, 4, 4);
                                    if(flag_segment){
                                        for(int i_2 = i_4; i_2 < i_4+4; i_2=i_2+2){
                                            for(int j_2 = j_4; j_2 < j_4+4;  j_2 = j_2+2){
                                                fillin(img_region, i_2, j_2, 2, 2, segment_value);
                                                segment_value++;
                                            }
                                        }
                                    }else{
                                        fillin(img_region, i_4, j_4, 4, 4, segment_value);
                                        segment_value++;
                                    }
                                }
                            }
                        }else{
                            fillin(img_region, i_8, j_8, 8, 8, segment_value);
                            segment_value++;
                        }
                    }
                }
            }else{
                fillin(img_region, i, j, 16, 16, segment_value);
                segment_value++;
            }
        }
    }

    cout<< segment_value<<endl;
    cout<<"-------------------------------------------------------------------------------------------"<<endl;
    display_img(img_region);
//    contain_one_count(img_edge);
//    cv::Mat img_region_display(size, CV_64FC1);
//    normalize(img_region, img_region_display, segment_value);
//    cv::namedWindow("sar");
//    cv::imshow("sar",img_region_display);
//    cv::waitKey(0);
    cv::Mat img_region_backup(img_region.size(), img_region.type());
    img_region.copyTo(img_region_backup);
    for(int i = 0 ;i< segment_value; i++){
        img_region_backup.copyTo(img_region);
        display_img(img_region);
        img_region = img_region - i;
        reverse_mat_value(img_region);
        display_img(img_region);
        cout<<endl;
        cout<<"----------------------------------------------------------------"<<endl;
    }


    cv::Mat test(3, 3, CV_8U);
    cv::Mat test_not(test.size(), test.type());
    init_mat_value(test, 1);
    display_img(test);

    test = test -1;
    test.at<uchar>(2, 2) = 3;
    reverse_mat_value(test);
    display_img(test);
    return 0;
}