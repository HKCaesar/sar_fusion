//
// Created by auroua on 16-6-16.
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

const double CONST_PI =       3.14159265358979323846;  /* pi */
const double CONST_PI_2 =     1.57079632679489661923;  /* pi/2 */
const double CONST_PI_4 =     0.78539816339744830962;  /* pi/4 */
const double CONST_PI_8 =     0.39269908169872415481;  /* pi/8 */
//const double THRESHOLD = 0.3;
//const double TR_THRESHOLE = 2.7;

const double THRESHOLD = 0.3;
const double TR_THRESHOLE = 2;


#define show_space() \
    std::cout<<"========================================================================="<<std::endl;

struct spixel{
    int x_left;
    int y_left;
    int height;
    int width;
    int weight;
};

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

void display_img_int(const cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        const int* datas = image.ptr<int>(i);
        for(int j = 0; j < image.cols; j++){
            cout<<(int)datas[j]<<", ";
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

void init_mat2(cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        uchar* datas = image.ptr<uchar>(i);
        for(int j = 0; j < image.cols; j++){
            datas[j] = 100;
        }
    }
}

void init_mat_double(cv::Mat& image){
//    cout << "in display_img rows="<<image.rows << " cols="<<image.cols<< endl;
    for(int i = 0; i < image.rows; i++){
        double* datas = image.ptr<double>(i);
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
            img_edge.at<double>(i, j) = values;
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

int contain_non_zero(const cv::Mat& img_edge){
    int value = 0;
    for(int i = 0; i<img_edge.rows; i++){
        for(int j=0; j<img_edge.cols; j++){
            if(img_edge.at<double>(i, j)>0){
//                cout<< "index of non-zero is ==="<< i << " ===="<<j<<endl;
                value++;
            }
        }
    }
//    cout<<value<<endl;
    return value;
}


int contain_non_zero_uchar(const cv::Mat& img_edge){
    int value = 0;
    for(int i = 0; i<img_edge.rows; i++){
        for(int j=0; j<img_edge.cols; j++){
            if(img_edge.at<uchar>(i, j)>0){
//                cout<< "index of non-zero is ==="<< i << " ===="<<j<<endl;
                value++;
            }
        }
    }
//    cout<<value<<endl;
    return value;
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
            temp = image.at<double>(i, j);
            if(temp!=0){
                image.at<double>(i, j) = 0;
            }else{
                image.at<double>(i, j) = 1;
            }
        }
    }
}

void fill_struct(spixel& pix, int x_l, int x_r, int h, int w, int value){
    pix.x_left = x_l;
    pix.y_left = x_r;
    pix.height = h;
    pix.width = w;
    pix.weight = value;
}

void show_map(map<int, spixel>& map_region){
    for(map<int, spixel>::iterator iElement = map_region.begin(); iElement!=map_region.end();++iElement){
        cout<<iElement->first<<" -> x_left: "<<iElement->second.x_left<<" y_left: "<<iElement->second.y_left
        <<" height: "<<iElement->second.height<<" width: "<<iElement->second.width<<" weight: "<<iElement->second.weight<<endl;
    }
}

void show_set(set<int> values){
    for(set<int>::iterator iElement = values.begin(); iElement!=values.end(); iElement++){
        cout<<*iElement<<" , ";
    }
    cout<<endl;
}

set<int> search_neghbors(const cv::Mat& imgs, const map<int, spixel>& regions, int weight){
    set<int> neghbors;
    spixel pix_info;
    map<int, spixel>::const_iterator pix_iter = regions.find(weight);
    pix_info = pix_iter->second;
    int rows = pix_info.x_left + pix_info.height;
    int cols = pix_info.y_left + pix_info.width;
    int pix_val = 0;
    for(int i=pix_info.x_left; i<(pix_info.x_left+pix_info.height); i++){
        pix_val = (int)imgs.at<double>(i, cols);
        if((pix_val-pix_info.weight)!=0){
            neghbors.insert(pix_val);
        }
    }
    for(int i=pix_info.y_left; i<(pix_info.y_left+pix_info.width); i++){
        pix_val = (int)imgs.at<double>(rows, i);
        if((pix_val-pix_info.weight)!=0){
            neghbors.insert(pix_val);
        }
    }
    return neghbors;
}


void copy_double_uchar(cv::Mat src, cv::Mat dst){
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            dst.at<uchar>(i,j) = src.at<double>(i, j);
        }
    }
}


cv::Mat generate_trval(const cv::Mat &img_region, const cv::Mat& img_input, int segmentVal){
    show_space();
    show_space();
    cv::Mat img_region1(img_region.size(), img_region.type());
    cv::Mat img_region2(img_region.size(), img_region.type());
    cv::Mat img_mask(img_region.size(), CV_8U);
    cv::Mat tr_val(segmentVal, segmentVal, CV_64FC1);
    init_mat_double(tr_val);
    double u0 = 0.0;
    double u01 = 0.0;
    double u02 = 0.0;
    double u1 = 0.0;
    double u2 = 0.0;
    double u4 = 0.0;
    double etasc = 0.0;
    double tr = 0.0;
    int N1 = 0;
    int N2 = 0;
    int N = 0;
    int n_covert = 0;

//    double tr_val[segmentVal][segmentVal];

    for(int i=0; i < segmentVal; i++){
        n_covert = 0;
        N1 = 0;
        u1 = 0;
        u01 = 0;
        init_mat(img_mask);

        img_region.copyTo(img_region1);
//        display_img_double(img_region1);
        // caculate region1 u1 and n1
        img_region1 = img_region1 - i;
//        display_img_double(img_region1);
        reverse_mat_value(img_region1);
//        display_img_double(img_region1);
        N1 = contain_non_zero(img_region1);
//        cout<< img_region.type()<<" ================================"<<img_region.size<<endl;
        copy_double_uchar(img_region1, img_mask);
//        display_img(img_mask);
        n_covert = contain_non_zero_uchar(img_mask);
        if(N1!=n_covert){
            cout<<"error error error error error error error error error error error error error error error error error error line 701";
        }
        u1 = cv::mean(img_input,img_mask)(0);
        u01 = u1*N1;

        for(int j=0; j<segmentVal; j++){
            if(i==j){
                continue;
            }
            img_region.copyTo(img_region2);
            init_mat(img_mask);
            u2 = 0.0;
            u02 = 0.0;
            u0 = 0.0;
            N = 0;
            N2 = 0;
            tr = 0.0;
            u4 = 0.0;
//            display_img_double(img_region2);
            img_region2 = img_region2 - j;
//            display_img_double(img_region2);
            reverse_mat_value(img_region2);
//            display_img_double(img_region2);
            N2 = contain_non_zero(img_region2);
//            cout<< img_region.type()<<" ================================"<<img_region.size<<endl;
            copy_double_uchar(img_region2, img_mask);
//            display_img(img_mask);
            n_covert = contain_non_zero_uchar(img_mask);
            if(N2!=n_covert){
                cout<<"error error error error error error error error error error error error error error error error error error line 701";
            }
//            display_img(img_input);
            u2 = cv::mean(img_input,img_mask)(0);
            u02 = u2*N2;
            u0 = u01 + u02;
            N = N1+N2;
            u4 = u0/N;
            tr = -N1*log(u1) - N2*log(u2) + (N1+N2)*log(u4);
            tr_val.at<double>(i, j) = tr;
        }
    }
//    return tr_val;
//    display_img_double(tr_val);
    return tr_val;
}




//2016-6-23
double square( double a )
{
    return a*a;
}


double diff( const cv::Mat &img, int x1, int y1, int x2, int y2 )
{
//    double result = img.at<double>( y1, x1 ) - img.at<double>( y2, x2 );
//    if(result){
//        cout<< "x1: "<< x1<<" y1: "<<y1<< "x2: "<< x2 <<" y2: "<< y2 <<" and the value is ==="<< result<< endl;
//    }
    return img.at<double>( y1, x1 ) - img.at<double>( y2, x2 );
}

struct UniverseElement
{
    int rank;
    int p;
    int size;

    UniverseElement() : rank( 0 ), size( 1 ), p( 0 ) {}
    UniverseElement( int rank, int size, int p ) : rank( rank ), size( size ), p( p ) {}
};


class Universe
{
private:
    vector<UniverseElement> elements;
    int num;

public:
    Universe( int num ) : num( num )
    {
        elements.reserve( num );

        for ( int i = 0; i < num; i++ )
        {
            elements.emplace_back( 0, 1, i );
        }
    }

    ~Universe() {}

    int find( int x )
    {
        int y = x;
        while ( y != elements[y].p )
        {
            y = elements[y].p;
        }
        elements[x].p = y;

        return y;
    }

    void join( int x, int y )
    {
        if ( elements[x].rank > elements[y].rank )
        {
            elements[y].p = x;
            elements[x].size += elements[y].size;
        }
        else
        {
            elements[x].p = y;
            elements[y].size += elements[x].size;
            if ( elements[x].rank == elements[y].rank )
            {
                elements[y].rank++;
            }
        }
        num--;
    }

    int size( int x ) const { return elements[x].size; }
    int numSets() const { return num; }
};


struct edge
{
    int a;
    int b;
    double w;
};

void show_edge_info(vector<edge>& values){
    for(vector<edge>::iterator iElement = values.begin(); iElement!=values.end(); iElement++){
        cout<<" the first point: "<<(*iElement).a<<" second point: "<<(*iElement).b
        <<" weight: "<<(*iElement).w<<endl;

    }
    cout<<endl;
}


bool operator<( const edge &a, const edge &b )
{
    return a.w < b.w;
}

//caculate two different pixel's similarity
shared_ptr<Universe> segmentGraph( int numVertices, int numEdges, vector<edge> &edges)
{
//    sort( edges.begin(), edges.end() );
    auto universe = make_shared<Universe>( numVertices );
    for ( auto &pedge : edges )
    {
        int a = universe->find( pedge.a );
        int b = universe->find( pedge.b );

        if ( a != b )
        {
            if (  pedge.w == 0  )
            {
                universe->join( a, b );
            }
        }
    }

    return universe;
}


// image segmentation using "Efficient Graph-Based Image Segmentation"
shared_ptr<Universe> segmentation( const cv::Mat &blurred)
{
    const int width = blurred.cols;
    const int height = blurred.rows;
    std::vector<edge> edges( width*height * 4 );

    //build edge relations
    int num = 0;
    int total = 0;
    for ( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            if ( x < width - 1 )
            {
                edges[num].a = y * width + x;
                edges[num].b = y * width + ( x + 1 );
                double result = diff( blurred, x, y, x + 1, y );
                edges[num].w = diff( blurred, x, y, x + 1, y );
                num++;
            }

            if ( y < height - 1 )
            {
                edges[num].a = y * width + x;
                edges[num].b = ( y + 1 ) * width + x;
                edges[num].w = diff( blurred, x, y, x, y + 1 );
                double result = diff( blurred, x, y, x + 1, y );
                num++;
            }

            if ( ( x < width - 1 ) && ( y < height - 1 ) )
            {
                edges[num].a = y * width + x;
                edges[num].b = ( y + 1 ) * width + ( x + 1 );
                edges[num].w = diff( blurred, x, y, x + 1, y + 1 );
                double result = diff( blurred, x, y, x + 1, y );
                num++;
            }

            if ( ( x < width - 1 ) && ( y > 0 ) )
            {
                edges[num].a = y * width + x;
                edges[num].b = ( y - 1 ) * width + ( x + 1 );
                edges[num].w = diff( blurred, x, y, x + 1, y - 1 );
                double result = diff( blurred, x, y, x + 1, y );

                num++;
            }
        }
    }
    auto universe = segmentGraph( width*height, num, edges);

    return universe;
}


//region operations
struct Region
{
    int size;
    int parent_label;
    int merged = 0;
    int processed = 0;
    cv::Rect rect;
    std::vector<int> labels;
    std::vector<cv::Point> boundaries_add;
    std::vector<cv::Point> boundaries_sub;
    std::vector<cv::Point> boundaries;
    std::vector<cv::Point> pixels;
    int total_pixel;
    Region() {}

    Region( const cv::Rect &rect, int label ) : rect( rect )
    {
        labels.push_back( label );
    }

    Region(
            const cv::Rect &rect, int size,
            const std::vector<int> &&labels
    )
            : rect( rect ), size( size ), labels( std::move( labels ) )
    {}

    Region& operator=( const Region& region ) = default;

    Region& operator=( Region&& region ) noexcept
    {
        if ( this != &region )
        {
            this->size = region.size;
            this->rect = region.rect;
            this->labels = std::move( region.labels );
        }

        return *this;
    }

    Region( Region&& region ) noexcept
    {
        *this = std::move( region );
    }
};


std::map<int, Region> extractRegions( const cv::Mat &img, std::shared_ptr<Universe> universe, set<int>& total_labels )
{
    std::map<int, Region> R;

    for ( int y = 0; y < img.rows; y++ )
    {
        for ( int x = 0; x < img.cols; x++ )
        {
            int label = universe->find( y*img.cols + x );
            total_labels.insert(label);
//            if ( R.find( label ) == R.end() )
//            {
//                R[label] = Region( cv::Rect( 100000, 100000, 0, 0 ), label );
//            }
            R[label].pixels.push_back(cv::Point(x, y));
//            if ( R[label].rect.x > x )
//            {
//                R[label].rect.x = x;
//            }
//
//            if ( R[label].rect.y > y )
//            {
//                R[label].rect.y = y;
//            }
//            // the bottom right corner
//            if ( R[label].rect.br().x < x )
//            {
//                R[label].rect.width = x - R[label].rect.x + 1;
//            }
//
//            if ( R[label].rect.br().y < y )
//            {
//                R[label].rect.height = y - R[label].rect.y + 1;
//            }
        }
    }

    return R;
}


using LabelRegion = std::pair<int, Region>;
using Neighbour = std::pair<int, int>;

bool isIntersecting( const Region &a, const Region &b )
{
    Region temp;
    temp.rect.x = a.rect.x;
    temp.rect.y = a.rect.y;
    temp.rect.width = a.rect.width + 1;
    temp.rect.height = a.rect.height + 1;
    return ( ( temp.rect & b.rect ).area() != 0 );
}

void visualize( const cv::Mat &img, std::shared_ptr<Universe> universe )
{
    const int height = img.rows;
    const int width = img.cols;
    std::vector<cv::Vec3b> colors;

    cv::Mat segmentated( height, width, CV_8UC3 );

    std::random_device rnd;
    std::mt19937 mt( rnd() );
    std::uniform_int_distribution<> rand256( 0, 255 );

    for ( int i = 0; i < height*width; i++ )
    {
        cv::Vec3b color( rand256( mt ), rand256( mt ), rand256( mt ) );
        colors.push_back( color );
    }

    for ( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            segmentated.at<cv::Vec3b>( y, x ) = colors[universe->find( y*width + x )];
        }
    }
//    cv::imwrite("/home/auroua/sassi.jpg", segmentated);
    cv::imshow( "Initial Segmentation Result", segmentated );
    cv::waitKey( 1 );
}


void visualize2( const cv::Mat &img, map<int, Region>& R, vector<int>& main_label )
{
    const int height = img.rows;
    const int width = img.cols;
    std::vector<cv::Vec3b> colors;

    cv::Mat segmentated( height, width, CV_8UC3 );

    std::random_device rnd;
    std::mt19937 mt( rnd() );
    std::uniform_int_distribution<> rand256( 0, 255 );

    for ( int i = 0; i < height*width; i++ )
    {
        cv::Vec3b color( rand256( mt ), rand256( mt ), rand256( mt ) );
        colors.push_back( color );
    }

    for(auto item = main_label.begin(); item!=main_label.end(); item++){
            for(auto inner_element=R[*item].pixels.begin(); inner_element!=R[*item].pixels.end();inner_element++){
                segmentated.at<cv::Vec3b>(*inner_element) = colors[*item];
            }
    }

    cv::imshow( "Initial Segmentation Result", segmentated );
    cv::waitKey( 0 );
}

shared_ptr<Universe> generateSegments( const cv::Mat &img)
{
    auto universe = segmentation(img);

//    visualize( img, universe );

    return universe;
}

struct CmpByKeyLength {
    bool operator()(const int k1, const int k2) {
        return k1 > k2;
    }
};


//2016-6-26
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


//add boundary detection methon
//v0.1 version only consider the regular rect area, this method tries to fix this bug, to fit the unregular rect like the following
//3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
//3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
void find_boundarys(map<int,Region>& R){
    for(auto iElement=R.begin(); iElement!=R.end();iElement++){
        for(auto item = iElement->second.pixels.begin(); item!=iElement->second.pixels.end(); item++){
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
                temp_flag2 = find_vec(iElement->second.pixels, temp2);
            }
            temp_flag = find_vec(iElement->second.pixels, temp);
            if(!temp_flag){
                iElement->second.boundaries_add.push_back(*item);
                iElement->second.boundaries.push_back(*item);
            }
            if(!temp_flag2){
                iElement->second.boundaries_sub.push_back(*item);
                iElement->second.boundaries.push_back(*item);
            }
        }
    }
}

void find_region_boundarys(Region& R){
    R.boundaries_add.clear();
    R.boundaries_sub.clear();
    R.boundaries.clear();
    for(auto item = R.pixels.begin(); item!=R.pixels.end(); item++){
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
                temp_flag2 = find_vec(R.pixels, temp2);
            }
            temp_flag = find_vec(R.pixels, temp);
            if(!temp_flag){
                R.boundaries_add.push_back(*item);
                R.boundaries.push_back(*item);
            }
            if(!temp_flag2){
                R.boundaries_sub.push_back(*item);
                R.boundaries.push_back(*item);
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

//    string input_url = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg";
    string input_url = "/home/auroua/backup/HB19929_016.jpg";
//    string input_url = "/home/auroua/workspace/matlab2015/mstar/MSTAR_Data/MSTAR_PUBLIC_MIXED_TARGETS_CD2/17_DEG/COL2/SCENE1/T62/HB19929.jpg";
//    string input_url = "/home/auroua/workspace/output16_16.png";
    string output_url = "/home/auroua/workspace/output2.png";
    cv::Mat img_input = cv::imread(input_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    int type = img_input.type();
    cv::Size size = img_input.size();
    cv::Mat img_output(size, type);
    cv::Mat img_touiz(size, CV_64FC1);
    cv::Mat img_edge(size, type);
    cv::Mat img_region(size, CV_64FC1);
    init_mat(img_output);
    init_mat(img_touiz);
    init_mat(img_edge);
    init_mat_double(img_region);
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
    map<int, spixel> regions;
    int segment_value = 0;
    bool flag_segment = false;
    spixel spix_val;
    for(int i=0; i< img_input.rows; i=i+16){
//        cout<<" index of image_input i value ==="<<i<<endl;
        for(int j=0;j<img_input.cols; j=j+16){
//            cout<<" index of image_input j value ==="<<j<<endl;
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
                                                fill_struct(spix_val, i_2, j_2, 2, 2, segment_value);
                                                regions.insert(make_pair(segment_value, spix_val));
                                                segment_value++;
                                            }
                                        }
                                    }else{
                                        fillin(img_region, i_4, j_4, 4, 4, segment_value);
                                        fill_struct(spix_val, i_4, j_4, 4, 4, segment_value);
                                        regions.insert(make_pair(segment_value, spix_val));
                                        segment_value++;
                                    }
                                }
                            }
                        }else{
                            fillin(img_region, i_8, j_8, 8, 8, segment_value);
                            fill_struct(spix_val, i_8, j_8, 8, 8, segment_value);
                            regions.insert(make_pair(segment_value, spix_val));
                            segment_value++;
                        }
                    }
                }
            }else{
                fillin(img_region, i, j, 16, 16, segment_value);
                fill_struct(spix_val, i, j, 16, 16, segment_value);
                regions.insert(make_pair(segment_value, spix_val));
                segment_value++;
            }
        }
    }

    cout<< segment_value<<endl;
    cout<<"-------------------------------------------------------------------------------------------"<<endl;
    display_img_double(img_region);

    cout<<"map value is================================================================================="<<endl;


    //2016-6-23  use region growing mehtod
    auto universe = generateSegments( img_region );
    Universe universe_back = *universe;

    int imgSize = img_input.total();
    list<int> total_label;
    list<int> temp_neghbors;
    set<int> total_labels_set;
    map<int, Region> R = extractRegions(img_region, universe,total_labels_set );
//    cout<< total_labels_set.size()<<endl;
//    cout<< "map size is ==="<<R.size()<<endl;
    map<int, int> find_table;

    for(auto iElement=R.begin(); iElement!=R.end(); iElement++){
        total_label.push_back(iElement->first);
    }
//    cout<<"total label size is ==="<<total_label.size()<<endl;

    find_boundarys(R);

    double u0 = 0.0;
    double u1 = 0.0;
    double u2 = 0.0;
    double u4 = 0.0;
    double tr = 0.0;
    int N1 = 0;
    int N2 = 0;
    int N = 0;
    int n_covert = 0;

    Region temp_1;
    Region temp_inner_1;
    cv::Point point1;
    cv::Point point2;
    int label;
    int label_inner;
    double sum;
    double sum_inner;
    vector<int> main_label;
    vector<int> processed;
    vector<int> merged;
    bool flag_merge = false;
    while(total_label.size()>0){
        N1 = 0;
        u1 = 0;
        sum = 0;
        flag_merge = false;
        label = total_label.front();
        processed.clear();
        temp_neghbors.clear();
        cout<< "outer label is"<< label << endl;
        if(R[label].processed==1){
            total_label.remove(label);
            continue;
        }
        if(R[label].merged==1){
            total_label.remove(label);
            continue;
        }
//
//        temp_1.rect.x = R[label].rect.x;
//        temp_1.rect.y = R[label].rect.y;
//        temp_1.rect.width = R[label].rect.width + 1;
//        temp_1.rect.height = R[label].rect.height + 1;
//        temp_neghbors.clear();

        N1 = R[label].pixels.size();
        for(auto items = R[label].pixels.begin(); items!= R[label].pixels.end(); items++){
            sum += (unsigned int)img_input.at<uchar>(*items);
        }
        u1 = sum/N1;
        main_label.push_back(label);

        for(auto iPixels = R[label].boundaries_add.begin(); iPixels!=R[label].boundaries_add.end();iPixels++){
            for ( auto a = R.cbegin(); a != R.cend(); a++ ){
                bool flag = false;
                if(a->first==label){
                    continue;
                }
                point1.x = iPixels->x+1;
                point1.y = iPixels->y+1;
                flag = find_vec(a->second.pixels, point1);
                if(flag){
                    auto location = find(temp_neghbors.begin(), temp_neghbors.end(), a->first);
                    if(location==temp_neghbors.end()){
                        temp_neghbors.push_back(a->first);
                    }
                }
            }
        }
        for(auto iPixels = R[label].boundaries_sub.begin(); iPixels!=R[label].boundaries_sub.end();iPixels++){
            for ( auto a = R.cbegin(); a != R.cend(); a++ ){
                bool flag = false;
                if(a->first==label){
                    continue;
                }
                point1.x = iPixels->x-1;
                point1.y = iPixels->y-1;
                flag = find_vec(a->second.pixels, point1);
                if(flag){
                    auto location = find(temp_neghbors.begin(), temp_neghbors.end(), a->first);
                    if(location==temp_neghbors.end()){
                        temp_neghbors.push_back(a->first);
                    }
                }
            }
        }

        while(temp_neghbors.size()>0){
            N2 = 0;
            u2 = 0;
            sum_inner = 0;
            label_inner = temp_neghbors.front();
            cout << "inner label is ==="<<label_inner<<endl;

            auto item_found = find(merged.cbegin(), merged.cend(), label_inner);
            if(item_found!=merged.cend()){
                temp_neghbors.remove(label_inner);
                continue;
            }
            if(R[label_inner].merged==1){
                temp_neghbors.remove(label_inner);
                continue;
            }
            if(R[label_inner].processed==1){
                temp_neghbors.remove(label_inner);
                continue;
            }

            if(label_inner==label){
                temp_neghbors.remove(label_inner);
                continue;
            }
//            temp_inner = R[label_inner];
            N2 = R[label_inner].pixels.size();
            for(auto items = R[label_inner].pixels.begin(); items!= R[label_inner].pixels.end(); items++){
                sum_inner += (unsigned int)img_input.at<uchar>(*items);
            }
            u2 = sum_inner/N2;
            u4 = (sum + sum_inner)/(N1+N2);
            tr = -N1*log(u1) - N2*log(u2) + (N1+N2)*log(u4);
            //merge
            if(tr < TR_THRESHOLE){
                temp_inner_1.rect.x = R[label_inner].rect.x;
                temp_inner_1.rect.y = R[label_inner].rect.y;
                temp_inner_1.rect.width = R[label_inner].rect.width + 1;
                temp_inner_1.rect.height = R[label_inner].rect.height + 1;

                //merge pixel to outer loop label
                for(auto items = R[label_inner].pixels.begin(); items!= R[label_inner].pixels.end(); items++){
                    R[label].pixels.push_back(*items);
                }
                R[label_inner].parent_label = label;
                R[label_inner].merged = 1;
                R[label_inner].processed = 1;

                merged.push_back(label_inner);
                total_label.remove(label_inner);
                temp_neghbors.remove(label_inner);
                flag_merge = true;
                //update outer region parmaters
                N1 = R[label].pixels.size();
                sum = 0;
                for(auto items = R[label].pixels.begin(); items!= R[label].pixels.end(); items++){
                    sum += (unsigned int)img_input.at<uchar>(*items);
                }
                u1 = sum/N1;
                total_label.remove(label_inner);
            }else{
                temp_neghbors.remove(label_inner);
                processed.push_back(label_inner);
            }

            if(temp_neghbors.size()==0&flag_merge){
                find_region_boundarys(R[label]);
                // add neghbors
                for(auto iPixels = R[label].boundaries_add.begin(); iPixels!=R[label].boundaries_add.end();iPixels++){
                    for ( auto a = R.cbegin(); a != R.cend(); a++ ){
                        bool flag = false;
                        if(a->first==label){
                            continue;
                        }
                        point1.x = iPixels->x+1;
                        point1.y = iPixels->y+1;
                        flag = find_vec(a->second.pixels, point1);
                        if(flag){
                            auto location = find(temp_neghbors.begin(), temp_neghbors.end(), a->first);
                            auto loc_merged = find(merged.begin(), merged.end(), a->first);
                            auto loc_processed = find(processed.begin(), processed.end(), a->first);
                            if((location==temp_neghbors.end())&(loc_merged==merged.end())&(loc_processed==processed.end())){
                                temp_neghbors.push_back(a->first);
                            }
                        }
                    }
                }
                for(auto iPixels = R[label].boundaries_sub.begin(); iPixels!=R[label].boundaries_sub.end();iPixels++){
                    for ( auto a = R.cbegin(); a != R.cend(); a++ ){
                        bool flag = false;
                        if(a->first==label){
                            continue;
                        }
                        point1.x = iPixels->x-1;
                        point1.y = iPixels->y-1;
                        flag = find_vec(a->second.pixels, point1);
                        if(flag){
                            auto location = find(temp_neghbors.begin(), temp_neghbors.end(), a->first);
                            auto loc_merged = find(merged.begin(), merged.end(), a->first);
                            auto loc_processed = find(processed.begin(), processed.end(), a->first);
                            if((location==temp_neghbors.end())&(loc_merged==merged.end())&(loc_processed==processed.end())){
                                temp_neghbors.push_back(a->first);
                            }
                        }
                    }
                }
            }

        }
        cout<<"label is ==="<<label<<" label elements is ====="<<R[label].pixels.size()<<endl;
        R[label].processed = 1;
        R[label].merged = 1;
        merged.push_back(label);
        total_label.remove(label);
    }

    multimap<int, int, CmpByKeyLength> final_data;
    show_space();
    int total_merged_val = 0;
    for(auto iElem=main_label.begin(); iElem!=main_label.end();iElem++){
        cout<<"the pixel value is ===" << *iElem<< " and the pixel size is ==="<<R[*iElem].pixels.size()<<endl;
        total_merged_val += R[*iElem].pixels.size();
        final_data.insert(make_pair(R[*iElem].pixels.size(),*iElem));
    }

    for (auto iter = final_data.begin(); iter != final_data.end(); ++iter) {
        cout << iter->first <<" ----- "<< iter->second << endl;
    }

//    cout << total_merged_val;
//    visualize2(img_input, R, main_label);
    auto iter = final_data.begin();
    iter++;
    int label_2 = iter->second;
    cv::Mat visual_mat(img_input.size(), img_input.type());
    init_mat2(visual_mat);
    for(auto iter_pixs=R[label_2].pixels.begin();iter_pixs!=R[label_2].pixels.end();iter_pixs++){
        visual_mat.at<uchar>(*iter_pixs) = img_input.at<uchar>(*iter_pixs);
    }

    iter++;
    label_2 = iter->second;
    for(auto iter_pixs=R[label_2].pixels.begin();iter_pixs!=R[label_2].pixels.end();iter_pixs++){
        visual_mat.at<uchar>(*iter_pixs) = img_input.at<uchar>(*iter_pixs);
    }

    iter++;
    label_2 = iter->second;
    for(auto iter_pixs=R[label_2].pixels.begin();iter_pixs!=R[label_2].pixels.end();iter_pixs++){
        visual_mat.at<uchar>(*iter_pixs) = img_input.at<uchar>(*iter_pixs);
    }

//    iter++;
//    iter++;
//    iter++;
//    iter++;
//    label_2 = iter->second;
//    for(auto iter_pixs=R[label_2].pixels.begin();iter_pixs!=R[label_2].pixels.end();iter_pixs++){
//        visual_mat.at<uchar>(*iter_pixs) = img_input.at<uchar>(*iter_pixs);
//    }
//    iter++;
//    label_2 = iter->second;
//    for(auto iter_pixs=R[label_2].pixels.begin();iter_pixs!=R[label_2].pixels.end();iter_pixs++){
//        visual_mat.at<uchar>(*iter_pixs) = img_input.at<uchar>(*iter_pixs);
//    }

    cv::imshow( "Initial Segmentation Result", visual_mat );
    cv::waitKey( 0 );
    return 0;
}