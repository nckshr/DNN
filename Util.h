#pragma once
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdio.h>
#include "lib/Eigen/Dense"
template <class T>
T getAt(std::list<T> source, int ix) {
	T toReturn;
	int i = 0;
	typename std::list<T>::iterator iter = source.begin();
	while (iter != source.end() && i < ix) {
		i++;
		++iter;
	}
	toReturn = (*iter);
	return toReturn;
};
template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
	assert(!(hi < lo));
	return (v < lo) ? lo : (hi < v) ? hi : v;
}
typedef std::pair<int, float> intfloatPair;
bool comparator(const intfloatPair& l, const intfloatPair& r);
std::vector<intfloatPair> sortWithIndexReturn(std::vector<float> toSort);
std::vector<intfloatPair> sortWithIndexReturn(std::list<float> toSort);
void softmax(std::vector<float> &x);
void softmax(Eigen::VectorXf &x);
void rotate2D(float &vx, float &vy, float theta);
void normalize(float &vx, float &vy);
int argmin(std::vector<float> &x);
int argmax(std::vector<float> &x);
int argmin(Eigen::VectorXf &x);
int argmax(Eigen::VectorXf &x);
std::vector<std::string> split_string(std::string s, char delim);