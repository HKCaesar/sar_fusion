//
// Created by auroua on 16-6-22.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;

void set_and(Mat input1, Mat input2, Mat output)//集合交运算
{
    int a = input1.rows;
    int b = input1.cols;
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < b; j++)
        {
            output.at<uchar>(i, j) = input1.at<uchar>(i, j) * input2.at<uchar>(i, j) / 255;
        }
    }
}

void set_or(Mat input1, Mat input2, Mat output)//集合并运算
{
    int a = input1.rows;
    int b = input1.cols;
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < b; j++)
        {
            if (input1.at<uchar>(i, j) == 0 && input2.at<uchar>(i, j) == 0)
            {
                output.at<uchar>(i, j) = 0;
            }
            else
            {
                output.at<uchar>(i, j) = 255;
            }
        }
    }
}

int equadj(Mat array1, Mat array2)//相等判断
{
    int a = 0;
    for (int i = 0; i < array1.rows; i++)
    {
        for (int j = 0; j < array1.cols; j++)
        {
            if (array1.at<uchar>(i, j) != array2.at<uchar>(i, j))
            {
                a = 1;
                break;
            }
        }
    }
    return a;
}

int main()
{
    Mat a = Mat::zeros(1, 10000, CV_16UC1);
    Mat kenel1 = Mat::ones(5, 5, CV_8UC1);
    Mat kenel2 = Mat::ones(3, 3, CV_8UC1);
    Mat img,img1;
    do
    {
        char m[100];
        cout << "请输入图片路径：" << endl;
        cin.getline(m, 100);
        img = imread(m, 0);
    } while (img.empty());
    img.copyTo(img1);
    //阈值处理
    for (int i = 0; i < img1.rows; i++)
    {
        for (int j = 0; j < img1.cols; j++)
        {
            if (img1.at<uchar>(i, j) > 235)
            {
                img1.at<uchar>(i, j) = 255;
            }
            else
            {
                img1.at<uchar>(i, j) = 0;
            }
        }
    }

    //腐蚀
    erode(img1, img1, kenel1, Point(-1, -1));
    //连通分量统计
    Mat img2 = Mat::zeros(img1.size(), CV_8UC1);
    Mat img3 = Mat::zeros(img1.size(), CV_8UC1);
    Mat img4 = Mat::zeros(img1.size(), CV_8UC1);
    int c = 1;
    for (int i = 0; i < img1.rows; i++)
    {
        for (int j = 0; j < img1.cols; j++)
        {
            if (img2.at<uchar>(i, j) != img1.at<uchar>(i, j))
            {
                img2.at<uchar>(i, j) = img1.at<uchar>(i, j);
                img4.at<uchar>(i, j) = img.at<uchar>(i, j);
                do
                {
                    dilate(img2, img2, kenel2, Point(-1, -1));
                    set_and(img1, img2, img2);
                    c = equadj(img2, img3);
                    if (c == 1)
                    {
                        img2.copyTo(img3);
                    }
                } while (c == 1);
            }
        }
    }
    //区域生长
    Mat img5 = Mat::zeros(img4.rows+2,img4.cols+2, CV_8UC1);
    Mat img6 = Mat::zeros(img5.size(), CV_8UC1);
    Mat img7 = Mat::zeros(img5.size(), CV_8UC1);
    Mat img8 = Mat::zeros(img5.size(), CV_8UC1);
    Mat img9 = Mat::zeros(img5.size(), CV_8UC1);
    for (int i = 1; i < img5.rows-1; i++)
    {
        for (int j = 1; j < img5.cols-1; j++)
        {
            img5.at<uchar>(i, j) = img.at<uchar>(i - 1, j - 1);//提取初始种子
        }
    }
    int e = 1;
    for (int i = 0; i < img4.rows; i++)
    {
        for (int j = 0; j < img4.cols - 1; j++)
        {
            if (img4.at<uchar>(i, j) != 0)
            {
                img6.at<uchar>(i + 1, j + 1) = 255;
                do
                {
                    for (int p = 0; p < img6.rows; p++)
                    {
                        for (int q = 0; q < img6.cols; q++)
                        {
                            if (img6.at<uchar>(p, q) != 0)
                            {
                                if (abs(img5.at<uchar>(p - 1, q - 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p - 1, q - 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p - 1, q - 1) = 0;
                                }
                                if (abs(img5.at<uchar>(p - 1, q) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p - 1, q) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p - 1, q) = 0;
                                }
                                if (abs(img5.at<uchar>(p -1, q + 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p - 1, q + 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p - 1, q + 1) = 0;
                                }
                                if (abs(img5.at<uchar>(p, q - 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p, q - 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p, q - 1) = 0;
                                }
                                if (abs(img5.at<uchar>(p, q + 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p, q + 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p, q + 1) = 0;
                                }
                                if (abs(img5.at<uchar>(p + 1, q - 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p + 1, q - 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p + 1, q - 1) = 0;
                                }
                                if (abs(img5.at<uchar>(p + 1, q) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p + 1, q) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p + 1, q) = 0;
                                }
                                if (abs(img5.at<uchar>(p + 1, q + 1) - img4.at<uchar>(i, j)) < 20)
                                {
                                    img6.at<uchar>(p + 1, q + 1) = 255;
                                }
                                else
                                {
                                    img6.at<uchar>(p + 1, q + 1) = 0;
                                }
                            }
                        }
                    }
                    e = equadj(img6, img7);
                    if (e == 1)
                    {
                        img6.copyTo(img7);
                    }
                } while (e == 1);
                set_or(img7, img8, img8);
                img9.copyTo(img6);
            }
        }
    }
    imwrite("F://区域生长.tif", img8);
    imshow("区域生长", img8);
    waitKey();
    return 0;
}
