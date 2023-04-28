#pragma once
#include "MIO.h"
#include "Sampling.h"
#include <vector>
#include <list>
#include <math.h>
#include "lib/Eigen/Dense"
#include <ostream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <time.h>
class DNNGradient;

class DNN {
public:
	int inputSize;
	std::vector<Eigen::MatrixXf> weights;
	std::vector<Eigen::VectorXf> prelus;
	std::vector < Eigen::VectorXf> biases;
	DNN();
	DNN(const char* modelFolder);
	void init();
	void addLayer(int size);
	void loadModel(const char* modelFolder);
	void saveModel(const char* modelFolder);
	void applyGradient(DNNGradient &gradient, float stepSize);
	Eigen::VectorXf forward(Eigen::VectorXf input);
	std::string print();
};
class DNNGradient {
public:
	std::vector<Eigen::VectorXf> dCost_dBiases;
	std::vector<Eigen::MatrixXf> dCost_dWeights;
	DNNGradient(DNN net) {
		dCost_dWeights = std::vector<Eigen::MatrixXf>(net.weights.size());
		dCost_dBiases = std::vector<Eigen::VectorXf>(net.weights.size());
		for (int w = 0; w < net.weights.size(); w++) {
			dCost_dWeights[w] = Eigen::MatrixXf::Zero(net.weights[w].rows(), net.weights[w].cols());
			dCost_dBiases[w] = Eigen::VectorXf::Zero(net.weights[w].rows());
		}
	}
	std::string print() {
		std::stringstream outStr;
		for (int i = 0; i < dCost_dWeights.size(); i++) {
			outStr << "-- GRADIENT LAYER " << std::setfill(' ') << std::setw(3) << i << " --" << std::endl;
			outStr << " W: " << std::endl;
			outStr << dCost_dWeights[i].format(CleanFmt) << std::endl;
			outStr << " B: " << std::endl;
			outStr << dCost_dBiases[i].format(CleanFmt) << std::endl;
		}
		return outStr.str();
	}
};
class CostFunction {
public:
	CostFunction() {		
	}
	virtual float getCost(Eigen::VectorXf &y, Eigen::VectorXf &yhat, Eigen::VectorXf &gradient) {
		std::cout << "WARNING! getCost being called in base class of CostFunction! Override this method!!";
		gradient = Eigen::VectorXf::Zero(y.size());
		return 0;
	}
};
class MSE: public CostFunction {
public:
	MSE():CostFunction() { }

	float getCost(Eigen::VectorXf &y, Eigen::VectorXf &yhat, Eigen::VectorXf &gradient) {
		Eigen::VectorXf diff = y - yhat;
		gradient = -2.0 * diff;
		return (diff.dot(diff));
	}	
};
class CatCrossEntropy : public CostFunction {
public:
	CatCrossEntropy() :CostFunction() { }
	
	float getCost(Eigen::VectorXf &y, Eigen::VectorXf &yhat, Eigen::VectorXf &gradient) {
		Eigen::VectorXf ySoft = yhat;
		softmax(ySoft);
		gradient = ySoft - y;
		int classIx = argmax(y);
		return -std::log(ySoft[classIx]);
	}
};
class MAE : public CostFunction {
public:
	MAE() :CostFunction() { }

	float getCost(Eigen::VectorXf &y, Eigen::VectorXf &yhat, Eigen::VectorXf &gradient) {
		Eigen::VectorXf diff = y - yhat;
		Eigen::VectorXf diff2 = yhat - y;
		gradient = (diff2).cwiseProduct((y.size()*diff2.cwiseAbs()).cwiseInverse());
		return (diff.dot(diff));
	}
};
class StaticCost : public CostFunction {
public:
	StaticCost() :CostFunction() { }
	float staticCost;
	Eigen::VectorXf staticGradient;
	void setCost(float f) {
		staticCost = f;
	}
	void setGradient(Eigen::VectorXf &gradient) {
		staticGradient = gradient;
	}
	float getCost(Eigen::VectorXf &y, Eigen::VectorXf &yhat, Eigen::VectorXf &gradient) {
		gradient = staticGradient;
		return staticCost;
	}
};
void train(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int numEpochs, int batchSize, float learningRate, float learningDecay, CostFunction* costFunc, std::string method);
void train(DNN &net, Eigen::MatrixXf &trainX, Eigen::MatrixXf &trainY, Eigen::MatrixXf &testX, Eigen::MatrixXf &testY, int numEpochs, int batchSize, float learningRate, float learningDecay, CostFunction* costFunc, std::string method);
void trainCEM(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int numEpochs, int batchSize, int poolSize, int winners, CostFunction* costFunc);
//at the epoch level, do things like adjust learning rate and shuffle batch order
float runEpoch(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int batchSize, float stepSize, CostFunction* costFunc);
float runEpochCEM(std::vector<DNN> &samples, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int batchSize, int winners, CostFunction* costFunc);
// at the batch level, sum gradients over the data and return the average as the gradient to be applied
float runBatch(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, CostFunction* costFunc, int batchSize, int offset, float stepSize);
float runBatchCEM(std::vector<DNN> &samples, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, CostFunction* costFunc, int batchSize, int offset, int winners);
// for a single input-ground_truth pair, compute the weight gradients and return average cost
float backProp(DNN &net, Eigen::VectorXf &input, Eigen::VectorXf &groundTruthOut, CostFunction* costFunc, DNNGradient &gradient, float gradientMultiplier);
//compute cost function
void feedForward(Eigen::VectorXf &inputs, Eigen::VectorXf &outputs, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus, int fromLayer, int toLayer, std::vector<Eigen::VectorXf> &intermediateValues);
Eigen::VectorXf feedForward(Eigen::VectorXf &inputs, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus, std::vector<Eigen::VectorXf> &intermediateValues);
Eigen::VectorXf feedForward(Eigen::VectorXf &inputs, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus);
void readModel(const char* modelFolder, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases);
void readModel(const char* modelFolder, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus);
void applyPrelu(Eigen::VectorXf &prelu, Eigen::VectorXf &vals);
void makeCEMSamples(DNN &model, std::vector<DNN> &agents, int poolSize);

