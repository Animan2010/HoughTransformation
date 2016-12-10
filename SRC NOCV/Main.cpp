#include <mpi.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define SHOW_WINDOWS true
#define PI 3.14159265

using namespace std;

int rankProc;
int countProc;
bool mainProc;
double t1, t2;

void houghLine(float accuracy = 0.1)
{
    char *bin = 0;
    int *phase = 0;

    int RMax;
    int w, h;

    if (mainProc) {
        cout << "FIRST PROC ENTER\n";

        ifstream file;
        file.open("img.txt", ios::in);
        file >> w;
        file >> h;
        bin = new char[w * h];
        for (int i = 0; i < w * h; i++)
            file >> bin[i];
        cout << "LOADED BINARY IMAGE\n";
        file.close();

        // максимальное расстояние от начала координат - это длина диагонали
        RMax = round(sqrt((double)(w*w + h*h)));
        // картинка для хранения фазового пространства Хафа (r, f)
        // 0 < r < RMax
        // 0 < f < 2*PI
        phase = new int[w * h];
        memset(phase, 0, sizeof(int) * w * h);

        t1 = MPI_Wtime();

        for (int i = 1; i < countProc; i++) {
            MPI_Send(&h, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // h - rows
            MPI_Send(&w, 1, MPI_INT, i, 1, MPI_COMM_WORLD); // w - cols
            MPI_Send(bin, w * h, MPI_CHAR, i, 2, MPI_COMM_WORLD);
            MPI_Send(&RMax, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            cout << "MAIN PROCESS SEND\n";
        }
    } else {
        MPI_Status status;

        MPI_Recv(&h, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&w, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        char* buf = new char[w * h];
        MPI_Recv(buf, w * h, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&RMax, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        
        bin = buf;
        phase = new int[w * h];
        memset(phase, 0, sizeof(int) * w * h);

        cout << "SUBPROC RECEIVE\n";
    }

    cout << "RECEIVED\n";

    int x = 0, y = 0, r = 0, f = 0;
    float theta = 0;

    // пробегаемся по пикселям изображения контуров
    for (y = rankProc; y < h; y += countProc) {\
        for (x = 0; x < w; x++) {
            int index = y * w + x;
            if (bin[index] == '1') { // это пиксель контура?
                            // рассмотрим все возможные прямые, которые могут 
                            // проходить через эту точку
                for (f = 0; f<180; f++) { //перебираем все возможные углы наклона
                    for (r = 0; r<RMax; r++) { // перебираем все возможные расстояния от начала координат
                        int phaseIndex = f * w + r;
                        theta = f * PI / 180.0; // переводим градусы в радианы
                                                 // Если решение уравнения достаточно хорошее (точность больше заданой)
                        if (abs(((y)*sin(theta) + (x)*cos(theta)) - r) < accuracy) {
                            phase[phaseIndex]++; // увеличиваем счетчик для этой точки фазового пространства.
                        }
                    }
                }
            }
        }
    }

    cout << "PHASE CREATED\n";

    // Выбираем точку фазового пространства которая набрала наибольшее число попаданий
    unsigned int MaxPhaseValue = 0;
    float Theta = 0;
    int R = 0;
    for (f = 0; f < 180; f++) { //перебираем все возможные углы наклона
        for (r = 0; r<RMax; r++) { // перебираем все возможные расстояния от начала координат
            if (phase[f * w + r] > MaxPhaseValue) {
                MaxPhaseValue = phase[f * w + r];
                Theta = f;
                R = r;
            }
        }
    }

    if (mainProc) {
        for (int i = 1; i < countProc; i++) {
            MPI_Status status;
            unsigned int phaseValue;
            float theta;
            int r;
            MPI_Recv(&phaseValue, 1, MPI_UNSIGNED, i, 10, MPI_COMM_WORLD, &status);
            MPI_Recv(&theta, 1, MPI_FLOAT, i, 11, MPI_COMM_WORLD, &status);
            MPI_Recv(&r, 1, MPI_INT, i, 12, MPI_COMM_WORLD, &status);
            if (phaseValue > MaxPhaseValue) {
                MaxPhaseValue = phaseValue;
                Theta = theta;
                R = r;
            }
        }
    } else {
        MPI_Send(&MaxPhaseValue, 1, MPI_UNSIGNED, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&Theta, 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
        MPI_Send(&R, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
    }

    cout << "FINISHED\n";

    if (mainProc) {
        // Рисуем линию по точкам для  R, Teta которые получили в результате преобразования
        Theta = Theta * PI / 180.0;
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                if (round((y * sin(Theta) + x * cos(Theta))) == R) {
                    bin[y * w + x] = '2';
                }
            }
        }
        t2 = MPI_Wtime();
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
    MPI_Comm_size(MPI_COMM_WORLD, &countProc);
    mainProc = (rankProc == 0);

    houghLine();

    if (mainProc) {
        printf("Time: %f\n", t2 - t1);
    }

    MPI_Finalize();

    return 0;
}