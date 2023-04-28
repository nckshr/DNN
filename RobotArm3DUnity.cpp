#include "DRL.h"
#include "DNN.h"
#include "Sampling.h"
#include "lib/Eigen/Dense"
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <iostream>
std::ofstream drl_log;
// Backup streambuffers of  cout 
std::streambuf* stream_buffer_cout;
std::streambuf* stream_buffer_cin;
int agentHandle;
extern "C" {
	// Function pointer to the C# function
	// The syntax is like this: ReturnType (*VariableName)(ParamType ParamName, ...)
	__declspec(dllexport) float (*takeActionUnity)(int handle, float* action, float* nextState);
	__declspec(dllexport) int (*getRandomStateUnity)(int handle, float* state);
	//__declspec(dllexport) int(*getRandomStateUnity)();
	// C++ function that C# calls
	// Takes the function pointer for the C# function that C++ can call
	__declspec(dllexport) void Init(int handle, float (*takeActionPtr)(int h, float* action, float* nextState), int(*getRandomStatePtr)(int h, float* state))
	//__declspec(dllexport) void Init(float(*takeActionPtr)(float* action, float* nextState), int(*getRandomStatePtr)())
	{
	
		takeActionUnity = takeActionPtr;
		getRandomStateUnity = getRandomStatePtr;
		stream_buffer_cout = std::cout.rdbuf();
		stream_buffer_cin = std::cin.rdbuf();
		agentHandle = handle;
		
	}
	
}
class RobotArm3DEnv : public DRLEnvironment{
public:
	int numLinks;
	RobotArm3DEnv() :DRLEnvironment() {
		numLinks = 1;
	}
	void setLinks(int l) {
		numLinks = l;
	}
	int getActionSize() {
		return numLinks;
	}
	int getStateSize() {
		return 6;
	}
	int trajectorySize;
	int getTrajectorySize() {
		return trajectorySize;
	}
	void getAction(Eigen::VectorXf &action, Eigen::VectorXf &policyOut, Eigen::VectorXf &policyGradient) {
		//policyOut = [m1,m2,m3...mn,log(c1),log(c2),log(c3)...log(cn)]
		policyGradient = Eigen::VectorXf(numLinks*2);
		for (int i = 0; i < numLinks; i++) {
			float sigma = std::exp(policyOut[i + numLinks]);
			action[i] = sampleGaussian(policyOut[i], sigma);
			//for CEM, output sigma directly
			//action[i] = sampleGaussian(policyOut[i], policyOut[i+numLinks]);
			//policyGradient[i] = (action[i] - policyOut[i]) / std::powf(sigma, 2.0);
			//policyGradient[i + numLinks] = policyGradient[i] * (action[i] - policyOut[i]);
		}
		//drl_log << "action from policy: " << action.format(CleanFmt) << std::endl;
		//drl_log << "policy gradient: " << policyGradient.format(CleanFmt) << std::endl;
	}
	float takeAction(Eigen::VectorXf &action, Eigen::VectorXf &state, Eigen::VectorXf &nextState) {
		//drl_log << "executing action " << action.format(CleanFmt) << std::endl;
		nextState = Eigen::VectorXf(getStateSize());
		float reward = takeActionUnity(agentHandle,action.data(), nextState.data());
		//drl_log << "next state: " << nextState.format(CleanFmt) << std::endl;
		return reward;
	}
	void sampleState(Eigen::VectorXf &state) {
	//	drl_log << "sampling a state..." << std::endl;
		int res = getRandomStateUnity(agentHandle, state.data());
		//drl_log << "successfully sampled a state" << state.format(CleanFmt) << std::endl;
	}

};
DNN learner;
std::vector<DNN> cemAgents;
RobotArm3DEnv env;
bool verbose;
int numLinks = 2;
extern "C"{
	__declspec(dllexport) void initLearner(int ts, int nHiddenLayers, int* hiddenLayerSizes) {
		env = RobotArm3DEnv();
		env.trajectorySize = ts;
		env.setLinks(numLinks);
		learner = DNN();
		drl_log << "initializing policy..." << std::endl;
		//input x, y, z, gx, gy, gz
		learner.addLayer(env.getStateSize());
		for (int l = 0; l < nHiddenLayers; l++) {
			learner.addLayer(hiddenLayerSizes[l]);
		}
		// numLinks means, numLinks sigmas
		learner.addLayer(env.numLinks*2);
		drl_log << "creating learner..." << std::endl;

	}
	__declspec(dllexport) void initCEM(int poolSize) {		
		cemAgents = std::vector<DNN>(poolSize);
		makeCEMSamples(learner,cemAgents, poolSize);
	}
	__declspec(dllexport) void trainLearner(int numBatches, int trajectoriesPerBatch, float stepSize, float stepDecay, float discount, float lambda) {
		drl_log << "training learner..." << std::endl;
		learn(learner, &env, numBatches, trajectoriesPerBatch, stepSize, stepDecay, discount, lambda);
	}
	__declspec(dllexport) void trainLearnerCEM(int numBatches, int trajectoriesPerBatch, int poolSize, int winners) {
		drl_log << "training learner CEM..." << std::endl;
		learnCEM(learner, &env, numBatches, trajectoriesPerBatch, poolSize, winners);
	}
	__declspec(dllexport) float doBatch(int numTrajectories, float stepSize, float discount, float lambda) {
		drl_log << "running batch..." << std::endl;
		float avgReward = runBatch(learner, &env, numTrajectories, stepSize, discount, lambda);
		return avgReward;
	}
	__declspec(dllexport) float doBatchCEM(int numTrajectories, int winners) {
		drl_log << "running batch CEM..." << std::endl;
		float avgReward = runBatchCEM(cemAgents, &env, numTrajectories, winners);
		learner = cemAgents[cemAgents.size() - 1];
		drl_log << "example agent: " << std::endl << cemAgents[0].print() << std::endl;
		return avgReward;
	}
	__declspec(dllexport) int getActionSize() {
		return env.numLinks;
	}
	__declspec(dllexport) void setLinks(int links) {
		numLinks = links;
	}
	__declspec(dllexport) void getAction(float* stateBuff, float *actionBuff) {
		//drl_log << "getting Action..." << std::endl;
		Eigen::VectorXf stateIn(env.getStateSize());
		for (int i = 0; i < stateIn.size(); i++) {
			stateIn[i] = stateBuff[i];
		}
		Eigen::VectorXf policyOut = learner.forward(stateIn);
		Eigen::VectorXf policyGradient;
		Eigen::VectorXf action(env.getActionSize());
		env.getAction(action, policyOut, policyGradient);
		for (int i = 0; i < action.size(); i++) {
			actionBuff[i] = action[i];
		}
	}	
	__declspec(dllexport) void setVerbose(bool b) {
		verbose = b;
		if (drl_log.is_open()) drl_log.close();
		drl_log = std::ofstream("DeepRL_log.txt");

		if (verbose) {
			// Get the streambuffer of the file 
			drl_log.rdbuf();

			// Redirect cout to file 
			std::cout.rdbuf(drl_log.rdbuf());

		}
		else {
			// Redirect cout back to screen 
			std::cout.rdbuf(stream_buffer_cout);			
		}

	}
}