#pragma once
#include "DNN.h"
#include "lib/Eigen/Dense"
#include "Sampling.h"
class DRLEnvironment{
public:
	DRLEnvironment() {

	}
	virtual void getAction(Eigen::VectorXf &state, Eigen::VectorXf &policyOut, Eigen::VectorXf &policyGradient) {
		return ;
	}
	virtual float takeAction(Eigen::VectorXf &action, Eigen::VectorXf &state, Eigen::VectorXf &nextState) {
		return 0;
	}	
	virtual void sampleState(Eigen::VectorXf &state) {
		return ;
	}
	virtual int getActionSize() {
		return 0;
	}
	virtual int getStateSize() {
		return 0;
	}
	virtual int getTrajectorySize() {
		return 0;
	}

};
float sampleTrajectory(DNN &net, DRLEnvironment* env, Eigen::VectorXf &state0, std::vector<Eigen::VectorXf> &actions, std::vector<Eigen::VectorXf> &states, std::vector<float> &rewardsToGo, DNNGradient &gradient, float scaleFactor, float discount, float lambda);
float sampleTrajectoryCEM(DNN &net, DRLEnvironment* env, Eigen::VectorXf &state0);
float learn(DNN &net, DRLEnvironment* dre, int numBatches, int episodesPerBatch, float stepSize, float stepDecay, float discount, float lambda);
float learnCEM(DNN &model, DRLEnvironment* dre, int numBatches, int episodesPerBatch, int poolSize, int winners);
float runBatch(DNN &net, DRLEnvironment* dre, int episodesPerBatch, float stepSize, float discount, float lambda);
float runBatchCEM(std::vector<DNN> &agents, DRLEnvironment* dre, int trajectoriesPerBatch, int winners);
