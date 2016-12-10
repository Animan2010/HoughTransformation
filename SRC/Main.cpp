#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv\cv.h>

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define SHOW_WINDOWS true

using namespace std;
using namespace cv;

int rankProc;
int countProc;
bool mainProc;
double t1, t2;

void houghLine(IplImage* original, float accuracy = 2.0)
{
    IplImage *src = 0, *rgb = 0;
    IplImage *bin = 0;
    IplImage *phase = 0;

    int RMax;

    if (mainProc) {
        cout << "FIRST PROC ENTER\n";

        src = cvCloneImage(original);
        // заведём цветную картинку
        rgb = cvCreateImage(cvGetSize(src), 8, 3);
        cvConvertImage(src, rgb, CV_GRAY2BGR);
        // бинарная картинка - для контуров
        bin = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        cvCanny(src, bin, 50, 200);
        // максимальное расстояние от начала координат - это длина диагонали
        RMax = cvRound(sqrt((double)(src->width*src->width + src->height*src->height)));
        // картинка для хранения фазового пространства Хафа (r, f)
        // 0 < r < RMax
        // 0 < f < 2*PI
        phase = cvCreateImage(cvSize(RMax, 180), IPL_DEPTH_16U, 1);
        cvZero(phase);

        t1 = MPI_Wtime();

        char* img = new char[bin->width * bin->height];
        for (int y = 0; y < bin->height; y += 1) {
            uchar* ptr = (uchar*)(bin->imageData + y * bin->widthStep);
            for (int x = 0; x < bin->width; x++) {
                if (ptr[x] > 0) { // это пиксель контура?
                    img[x + y * bin->width] = '1';
                }
                else {
                    img[x + y * bin->width] = '0';
                }
            }
        }

        cout << "CONVERTED\n";

        for (int i = 1; i < countProc; i++) {
            MPI_Send(&(bin->width), 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&(bin->height), 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(img, bin->width * bin->height, MPI_CHAR, i, 2, MPI_COMM_WORLD);
            MPI_Send(&RMax, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            cout << "MAIN PROCESS SEND\n";
        }
    } else {
        MPI_Status status;
        int rows;
        int cols;

        MPI_Recv(&cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        char* buf = new char[rows * cols];
        MPI_Recv(buf, rows * cols, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&RMax, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

        bin = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
        for (int y = 0; y < rows; y++) {
            uchar* ptr = (uchar*)(bin->imageData + y * bin->widthStep);
            for (int x = 0; x < cols; x++) {
                if (buf[x + y * cols] == '1') {
                    ptr[x] = 1;
                }
                else {
                    ptr[x] = 0;
                }
            }
        }

        phase = cvCreateImage(cvSize(RMax, 180), IPL_DEPTH_16U, 1);
        cvZero(phase);

        cout << "SUBPROC RECEIVE\n";
    }

    cout << "RECEIVED\n";

    int x = 0, y = 0, r = 0, f = 0;
    float theta = 0;

    ofstream file;
    file.open("img.txt", ios::out);
    file << bin->width << ' ' << bin->height << ' ';
    for (y = rankProc; y < bin->height; y += countProc) {
        uchar* ptr = (uchar*)(bin->imageData + y * bin->widthStep);
        for (x = 0; x < bin->width; x++) {
            if (ptr[x] > 0) { // это пиксель контура?
                file << '1' << ' ';
            }
            else {
                file << '0' << ' ';
            }
        }
    }

    // пробегаемся по пикселям изображения контуров
    for (y = rankProc; y < bin->height; y += countProc) {
        uchar* ptr = (uchar*)(bin->imageData + y * bin->widthStep);
        for (x = 0; x<bin->width; x++) {
            if (ptr[x]>0) { // это пиксель контура?
                            // рассмотрим все возможные прямые, которые могут 
                            // проходить через эту точку
                for (f = 0; f<180; f++) { //перебираем все возможные углы наклона
                    short* ptrP = (short*)(phase->imageData + f * phase->widthStep);
                    for (r = 0; r<RMax; r++) { // перебираем все возможные расстояния от начала координат
                        theta = f*CV_PI / 180.0; // переводим градусы в радианы
                                                 // Если решение уравнения достаточно хорошее (точность больше заданой)
                        if (abs(((y)*sin(theta) + (x)*cos(theta)) - r) < accuracy) {
                            ptrP[r]++; // увеличиваем счетчик для этой точки фазового пространства.
                        }
                    }
                }
            }
        }
    }

    cout << "PHASE CREATED\n";

    // Выбираем точку фазового пространства которая набрала наибольшее число попаданий
    int MaxPhaseValue = 0;
    float Theta = 0;
    int R = 0;
    for (f = 0; f<180; f++) { //перебираем все возможные углы наклона
        short* ptrP = (short*)(phase->imageData + f * phase->widthStep);
        for (r = 0; r<RMax; r++) { // перебираем все возможные расстояния от начала координат
            if (ptrP[r]>MaxPhaseValue) {
                MaxPhaseValue = ptrP[r];
                Theta = f;
                R = r;
            }
        }
    }

    if (mainProc) {
        for (int i = 1; i < countProc; i++) {
            MPI_Status status;
            int phaseValue;
            float theta;
            int r;
            MPI_Recv(&phaseValue, 1, MPI_INT, i, 10, MPI_COMM_WORLD, &status);
            MPI_Recv(&theta, 1, MPI_FLOAT, i, 11, MPI_COMM_WORLD, &status);
            MPI_Recv(&r, 1, MPI_INT, i, 12, MPI_COMM_WORLD, &status);
            if (phaseValue > MaxPhaseValue) {
                MaxPhaseValue = phaseValue;
                Theta = theta;
                R = r;
            }
        }
    } else {
        MPI_Send(&MaxPhaseValue, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&Theta, 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
        MPI_Send(&R, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }

    cout << "FINISHED\n";

    if (mainProc) {
        // Рисуем линию по точкам для  R, Teta которые получили в результате преобразования
        Theta = Theta * CV_PI / 180.0;
        for (y = 0; y < rgb->height; y++) {
            uchar* ptr = (uchar*)(rgb->imageData + y * rgb->widthStep);
            for (x = 0; x < rgb->width; x++) {
                if (cvRound((y * sin(Theta) + x * cos(Theta))) == R) {
                    ptr[3 * x] = 0;
                    ptr[3 * x + 1] = 255;
                    ptr[3 * x + 2] = 255;
                }
            }
        }
        t2 = MPI_Wtime();
    }

    if (SHOW_WINDOWS && mainProc) {
        cvNamedWindow("bin", 1);
        cvShowImage("bin", bin);

        cvNamedWindow("line", 1);
        cvShowImage("line", rgb);
    }
    
    if (mainProc) {
        cvReleaseImage(&src);
        cvReleaseImage(&rgb);
        cvReleaseImage(&bin);
        cvReleaseImage(&phase);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
    MPI_Comm_size(MPI_COMM_WORLD, &countProc);
    mainProc = (rankProc == 0);

    IplImage *original = 0;

    if (mainProc) {
        // имя картинки задаётся первым параметром
        char* filename = "img.bmp";
        // получаем картинку
        original = cvLoadImage(filename);
    }

    houghLine(original);

    if (SHOW_WINDOWS && mainProc) {
        cvNamedWindow("original", 1);
        cvShowImage("original", original);
    }

    if (mainProc) {
        printf("Time: %f\n", t2 - t1);

        cvWaitKey(0);
        cvReleaseImage(&original);
        cvDestroyAllWindows();
    }

    MPI_Finalize();

    return 0;
}