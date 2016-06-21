#include <QApplication>

#include "mainwindow.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "mstar/read_data.h"
#include "Histogram1D.h"


using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();
//    return a.exec();
/* Extract Input & Output filenames */
    const char* MSTARname = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352.000";
    const char* JPEGname = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg";
    unsigned char* sar_magnitude = read_mstar(MSTARname, JPEGname, 1, 1, 1, 75);
//    cout<< (int)sar_magnitude[0] <<",  "<< (int)sar_magnitude[3] <<endl;
    int h = (unsigned int)sar_magnitude[0];
    int w = (unsigned int)sar_magnitude[1];
    Mat sar_magnitude_val(h, w, CV_8U);
    for(int i = 0; i< h; i++){
        for(int j = 0; j< w; j++){
            sar_magnitude_val.at<uchar>(i,j) = sar_magnitude[i*j+2];
        }
    }

//    Mat sar_magnitude_val = cv::imread("/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03352_000.jpg");
    Histogram1D hist;
    Mat histImg = hist.getHistogramImage(sar_magnitude_val);
    namedWindow("Histogram");
    imshow("Histogram", histImg);
    waitKey(0);
    cv::destroyAllWindows();
    return 0;
}