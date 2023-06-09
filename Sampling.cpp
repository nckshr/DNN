#include "Sampling.h"
#define ROOT2 1.41421356237 
#include <random>
std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0, 1.0);

float sampleGaussian(float mu, float sigma) {	
	return mu  + distribution(generator) * sigma;
	//float r = rand() / (float)RAND_MAX;
	//return ROOT2 * sigma * erfinvf(2*r-1.0) + mu;
}
float sampleWeibull(float k, float gamma) {
	float r = rand() / (float)RAND_MAX;
	return gamma*std::pow(-std::log(1 - r), 1.0 / k);
}
float sampleExponential(float lambda) {
	float r = rand() / (float)RAND_MAX;
	return -std::log(1 - r) / lambda;
}
int sample(std::vector<float> &probMass) {
	//probMass is assumed to sum to 1
	int ix = -1;
	float val = rand() / (float)RAND_MAX;
	//std::cout << "val: " << val << std::endl;
	float sum = 0;
	for (int i = 0; i < probMass.size(); i++) {
		sum += probMass[i];
		//std::cout << "sum: " << i << std::endl;
		if (sum > val) {
			//std::cout << "setting ix to " << i << std::endl;
			ix = i;
			break;
		}
	}
	return ix;
}

/* compute inverse error functions with maximum error of 2.35793 ulp */
float erfinvf(float a)
{
	float p, r, t;
	t = fmaf(a, 0.0f - a, 1.0f);
	t = std::log(t);
	if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
		p = 3.03697567e-10f; //  0x1.4deb44p-32 
		p = fmaf(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26 
		p = fmaf(p, t, 1.22150334e-6f); //  0x1.47e512p-20 
		p = fmaf(p, t, 2.84108955e-5f); //  0x1.dca7dep-16 
		p = fmaf(p, t, 3.93552968e-4f); //  0x1.9cab92p-12 
		p = fmaf(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9 
		p = fmaf(p, t, 4.83185798e-3f); //  0x1.3ca920p-8 
		p = fmaf(p, t, -2.64646143e-1f); // -0x1.0eff66p-2 
		p = fmaf(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1 
	}
	else { // maximum ulp error = 2.35456
		p = 5.43877832e-9f;  //  0x1.75c000p-28 
		p = fmaf(p, t, 1.43286059e-7f); //  0x1.33b458p-23 
		p = fmaf(p, t, 1.22775396e-6f); //  0x1.49929cp-20 
		p = fmaf(p, t, 1.12962631e-7f); //  0x1.e52bbap-24 
		p = fmaf(p, t, -5.61531961e-5f); // -0x1.d70c12p-15 
		p = fmaf(p, t, -1.47697705e-4f); // -0x1.35be9ap-13 
		p = fmaf(p, t, 2.31468701e-3f); //  0x1.2f6402p-9 
		p = fmaf(p, t, 1.15392562e-2f); //  0x1.7a1e4cp-7 
		p = fmaf(p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
		p = fmaf(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1 
	}
	r = a * p;
	return r;
}
