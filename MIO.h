#pragma once
#include "lib/Eigen/Dense"
#include <fstream>
#include <ostream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <list>
#include "Util.h"
// item sep, rowsep, row brackets
const Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
const Eigen::IOFormat RowFmt(4, 0, ", ", ";", "", "");
const Eigen::IOFormat SerialFmt(4, 0, " ", "\n", "", "");
const Eigen::IOFormat ListFmt(4, 0, ",", ",", "", "");
Eigen::MatrixXf matrixFromFile(const char* filename, int skip, char delim=' ');
Eigen::VectorXf vectorFromFile(const char* filename,bool lenFromHeader);
void matrixToFile(const char* filename, Eigen::MatrixXf data);
void vectorToFile(const char* filename, Eigen::VectorXf data);
std::ofstream openFile(const char* file, char mode);