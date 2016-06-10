#include <QApplication>

#include "mainwindow.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "mstar/read_data.h"

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
    read_mstar(MSTARname, JPEGname, 1, 1, 1, 75);
}
