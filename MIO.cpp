#pragma once
#include "MIO.h"
std::ofstream openFile(const char* file, char mode) {
	std::ofstream f;
	if (mode == 'w') {
		f = std::ofstream(file, std::ofstream::out);
	}
	else if (mode == 'r') {
		f = std::ofstream(file, std::ofstream::in);
	}
	if (f.fail()) {
		std::cout << "FAILURE: File path not found: " << file << std::endl;
		std::exit(1);
	}
	return f;
	
}

Eigen::MatrixXf matrixFromFile(const char* filename, int skip, char delim) {
	//std::cout << "Loading Matrix from file" << std::endl;
	Eigen::MatrixXf result;
	std::string line;
	
	std::fstream f(filename);	
	std::list<std::list<float>> rowList;
	
	int linecount = 0;
	int cols = 0;
	while (std::getline(f, line)) {
		if (skip > 0) {
			skip--; 
			continue;
		}
		std::vector<std::string> tokens = split_string(line, delim);
		std::list<float> row;	
		
		for (int i = 0; i < tokens.size(); i++) {
			float val = std::atof(tokens[i].c_str());
			row.push_back(val);
		}
		rowList.push_back(row);
		cols = row.size() > cols ? row.size(): cols;
		linecount++;
		//if (linecount % 1000 == 0) { std::cout << "read " << linecount << " rows..." << std::endl; }
	}
	int rix = 0;
	result = Eigen::MatrixXf(linecount, cols);
	for (std::list<std::list<float>>::iterator liter = rowList.begin(); liter != rowList.end(); ++liter) {
		int cix = 0;
		for (std::list<float>::iterator diter = (*liter).begin(); diter != (*liter).end(); ++diter) {
			result(rix, cix) = (*diter);
			cix++;
		}
		rix++;
	}
	return result;
}
Eigen::VectorXf vectorFromFile(const char* filename, bool lenFromHeader) {
	FILE* filepoint;
	int length = 0;
	errno_t err = fopen_s(&filepoint, filename, "r");
	if ( err != 0 ) {
		std::cout << "WARNING: Target file " << filename << " not found!! Result will be empty vector!!" << std::endl;
	}
	fclose(filepoint);
	Eigen::VectorXf result;
	std::string line;
	std::fstream f(filename);
	if (lenFromHeader) {
		std::getline(f, line);
		std::stringstream ss(line);
		ss >> length;
	}
	else {
		while (std::getline(f, line)) {
			length++;
		}
		f.close();
		f = std::fstream(filename);
	}
	result = Eigen::VectorXf(length);
	int linecount = 0;
	while (std::getline(f, line)) {
		std::stringstream ss(line);
		float val;
		ss >> val;
		result[linecount] = val;
		linecount++;
	}
	return result;
}
void matrixToFile(const char* filename, Eigen::MatrixXf data) {
	FILE* filepoint;
	errno_t err = fopen_s(&filepoint, filename, "w");
	if (err != 0) {
		std::cout << "ERROR: Could not open file at " << filename << std::endl;
	}
	std::ofstream fout(filepoint);
	fout << data.format(SerialFmt) << std::endl;
	fout.close();
	fclose(filepoint);

}
void vectorToFile(const char* filename, Eigen::VectorXf data) {
	FILE* filepoint;
	errno_t err = fopen_s(&filepoint, filename, "w");
	if (err != 0) {
		std::cout << "ERROR: Could not open file at " << filename << std::endl;
	}
	std::ofstream fout(filepoint);
	fout << data.format(SerialFmt)<< std::endl;
	fout.close();
	fclose(filepoint);
}

