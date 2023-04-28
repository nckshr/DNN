#include "Util.h"
bool comparator(const intfloatPair& l, const intfloatPair& r)
{
	return l.second < r.second;
}
void rotate2D(float &vx, float &vy, float theta) {
	float cosa = cos(theta), sina = sin(theta);
	//rotate goal heading by angle to get this heading
	float vxtemp = vx*cosa + vy*-sina; 
	float vytemp = vx*sina + vy*cosa;
	vx = vxtemp;
	vy = vytemp;
}
std::vector<intfloatPair> sortWithIndexReturn(std::vector<float> toSort) {
	std::vector<intfloatPair> sorted(toSort.size());
	for (int i = 0; i < toSort.size(); i++) {
		sorted[i] = intfloatPair(i, toSort[i]);
	}
	std::sort(sorted.begin(),sorted.end(),comparator);
	return sorted;
}
std::vector<intfloatPair> sortWithIndexReturn(std::list<float> toSort) {
	std::vector<intfloatPair> sorted(toSort.size());
	int ix = 0;
	for (std::list<float>::iterator iter = toSort.begin(); iter != toSort.end(); ++iter) {
		sorted[ix] = intfloatPair(ix, (*iter));
		ix++;
	}	
	std::sort(sorted.begin(), sorted.end(),comparator);
	return sorted;
}
void softmax(std::vector<float> &x){
	float esum = 0;
	for (int i = 0; i < x.size(); i++) {
		x[i] = std::exp(x[i]);
		esum += x[i];
	}
	for (int i = 0; i < x.size(); i++) {
		x[i] = x[i] / esum;
	}
}
void softmax(Eigen::VectorXf &x) {
	float esum = 0;
	for (int i = 0; i < x.size(); i++) {
		x[i] = std::exp(x[i]);
		esum += x[i];
	}
	for (int i = 0; i < x.size(); i++) {
		x[i] = x[i] / esum;
	}
}
int argmin(std::vector<float> &x) {
	int amin = -1;
	float min = 9e99;
	for (int i = 0; i < x.size(); i++) {
		if (x[i] < min) {
			amin = i;
			min = x[i];
		}
	}
	return amin;
}
int argmax(std::vector<float> &x) {
	int m = 0;
	float maxVal = -9e99;
	//std::cout << "Logits: " << std::endl;	
	for (int i = 0; i < x.size(); i++) {
		//std::cout << layerin[i] << ", ";
		if (x[i] > maxVal) {
			maxVal = x[i];
			m = i;
		}
	}
	//std::cout << std::endl;
	return m;
}
int argmin(Eigen::VectorXf &x) {
	int amin = -1;
	float min = 9e99;
	for (int i = 0; i < x.size(); i++) {
		if (x[i] < min) {
			amin = i;
			min = x[i];
		}
	}
	return amin;
}
int argmax(Eigen::VectorXf &x) {
	int m = 0;
	float maxVal = -9e99;
	//std::cout << "Logits: " << std::endl;	
	for (int i = 0; i < x.size(); i++) {
		//std::cout << layerin[i] << ", ";
		if (x[i] > maxVal) {
			maxVal = x[i];
			m = i;
		}
	}
	//std::cout << std::endl;
	return m;
}
void normalize(float &vx, float &vy) {
	float norm = std::sqrt(vx*vx + vy*vy);
	vy = vy / norm;
	vx = vx / norm;
}
std::vector<std::string> split_string(std::string s, char delim) {
	std::istringstream ss(s);

	std::vector<std::string> parts;
	std::string part;
	while (std::getline(ss, part, delim)) {
		if (!part.empty()) {
			parts.push_back(part);
		}
	}

	return parts;
}