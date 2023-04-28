#pragma once
#include <vector>
#include <list>
#include <algorithm>
#include <string>

float erfinvf(float a);
float sampleGaussian(float m, float c);
float sampleWeibull(float k, float gamma);
float sampleExponential(float lambda);
int sample(std::vector<float> &probMass);

