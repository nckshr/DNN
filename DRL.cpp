#include "DRL.h"

float learn(DNN &net, DRLEnvironment* dre, int numBatches, int trajectoriesPerBatch, float stepSize, float stepDecay, float discount, float lambda) {
	srand(time(NULL));
	float step = stepSize;
	float avgAvgCost = 0;
	for (int i = 0; i < numBatches; i++) {
		float avgCost = runBatch(net,dre,trajectoriesPerBatch,step, discount, lambda);
		avgAvgCost += avgCost;
		std::cout << "\n -- BATCH " << std::setfill(' ') << std::setw(5) << i << " / " << numBatches << " -- \n Average Batch Reward To Go: " << std::setfill(' ') << std::setw(10) << avgCost << std::endl;
		step = step * stepDecay;
	}
	return avgAvgCost / (int)numBatches;
}
float learnCEM(DNN &model, DRLEnvironment* dre, int numBatches, int trajectoriesPerBatch, int poolSize, int winners) {
	srand(time(NULL));
	std::vector<DNN> agents(poolSize);
	makeCEMSamples(model, agents, poolSize);
	float avgAvgCost = 0;
	for (int i = 0; i < numBatches; i++) {
		float avgCost = runBatchCEM(agents, dre, trajectoriesPerBatch, winners);
		avgAvgCost += avgCost;
		std::cout << "\n -- BATCH " << std::setfill(' ') << std::setw(5) << i << " / " << numBatches << " -- \n Average Batch Reward To Go: " << std::setfill(' ') << std::setw(10) << avgCost << std::endl;
	}
	//output last sampled agent
	model = agents[agents.size()];
	return avgAvgCost / (int)numBatches;
}
float runBatchCEM(std::vector<DNN> &agents, DRLEnvironment* dre, int trajectoriesPerBatch,int winners) {
	std::cout << "running CEM batch... " << std::endl;
	int nlayers = agents[0].weights.size();
	std::vector<float>rewards(agents.size());	
	Eigen::VectorXf state0(dre->getStateSize());	
	//evaluate each network
	float avgPoolAvgReward = 0;
	std::cout << "evaluating agents... " << std::endl;
	for (int i = 0; i < trajectoriesPerBatch; i++) {
		std::cout << "sampling state... " << std::endl;
		dre->sampleState(state0);
		//get reward for each agent
		float agentAvgReward = 0;
		for (int j = 0; j < agents.size(); j++) {
			rewards[j] += sampleTrajectoryCEM(agents[j], dre, state0) / (float)trajectoriesPerBatch;
			avgPoolAvgReward += rewards[j] / (float)agents.size();
		}		
	}
	float avgRewardVar = 0;
	for (int j = 0; j < agents.size(); j++) {
		avgRewardVar += std::powf(rewards[j] - avgPoolAvgReward,2) / (float)agents.size();
	}
	std::cout << "stats of average rewards: " << avgPoolAvgReward << " +/- " << avgRewardVar << std::endl;
	std::cout << "building distributions... " << std::endl;
	//now we have average agent rewards on the same trajectories. Collect best of the bunch and resample new set of agents
	std::vector<std::vector<float>>weightMeans(nlayers);
	std::vector<std::vector<float>>biasMeans(nlayers);
	std::vector<std::vector<float>>weightVars(nlayers);
	std::vector<std::vector<float>>biasVars(nlayers);
	for (int l = 0; l < nlayers; l++) {
		int rows = agents[0].weights[l].rows();
		int cols = agents[0].weights[l].cols();
		int b = agents[0].biases[l].size();
		weightMeans[l] = std::vector<float>(rows*cols,0);
		weightVars[l] = std::vector<float>(rows*cols,0);
		biasMeans[l] = std::vector<float>(b,0);
		biasVars[l] = std::vector<float>(b,0);
	}
	std::cout << "collecting best agents... " << std::endl;
	float totalRewards = 0;
	//TODO: Replace this with min-heap for speed
	//you really want to do min-heap for this...
	//will sort increasing
	std::vector<intfloatPair> sortedRewards = sortWithIndexReturn(rewards);
	//take the top x % of networks and update parameter gaussians
	//compute sample means & stdevs
	//n-1 for sample distribution (because stats)
	std::cout << "building sample distributions... " << std::endl;
	float scale = (1.0f / (float)(winners - 1));
	//std::cout << "winner rewards: ";
	for (int i = agents.size()-1; i > agents.size()-winners; i--) {
		totalRewards += sortedRewards[i].second;
		int ix = sortedRewards[i].first;
		//std::cout << sortedRewards[i].second << ", ";
		for (int l = 0; l < agents[ix].weights.size(); l++) {
			int rows = agents[ix].weights[l].rows();
			int cols = agents[ix].weights[l].cols();
			float* wdata = agents[ix].weights[l].data();
			for (int w = 0; w < rows*cols; w++) {
				weightMeans[l][w] += wdata[w] * scale;		
			}
			for (int b = 0; b < rows; b++) {
				biasMeans[l][b] += agents[ix].biases[l][b] * scale;
			}
			
		}
		for (int l = 0; l < agents[ix].weights.size(); l++) {
			int rows = agents[ix].weights[l].rows();
			int cols = agents[ix].weights[l].cols();
			float* wdata = agents[ix].weights[l].data();

			for (int w = 0; w < rows*cols; w++) {
				weightVars[l][w] += std::powf(wdata[w] - weightMeans[l][w],2.0) * scale;
			}
			for (int b = 0; b < rows; b++) {
				biasVars[l][b] += std::powf(agents[ix].biases[l][b] - biasMeans[l][b],2.0) * scale;
			}

		}		
	}	
	std::cout << std::endl << "re-sampling... " << std::endl;
	//sample from these to get new networks
	for (int i = 0; i < agents.size(); i++) {
		for (int l = 0; l < agents[i].weights.size(); l++) {
			int rows = agents[i].weights[l].rows();
			int cols = agents[i].weights[l].cols();
			float* wdata = agents[i].weights[l].data();
			for (int w = 0; w < rows*cols; w++) {
				wdata[w] = sampleGaussian(weightMeans[l][w],std::powf(weightVars[l][w],0.5f));
			}
			for (int b = 0; b < rows; b++) {
				agents[i].biases[l][b] = sampleGaussian(biasMeans[l][b], std::powf(biasVars[l][b],0.5f));
			}
		}
	}
	float avgAvgReward = totalRewards / (float)agents.size();
	std::cout << "average agent average reward: " << avgAvgReward << std::endl;
	return avgAvgReward;
}
float runBatch(DNN &net, DRLEnvironment* dre, int trajectoriesPerBatch, float stepSize, float discount, float lambda) {
	int trajsize = dre->getTrajectorySize();
	std::vector<Eigen::VectorXf> actions(trajsize);
	std::vector<Eigen::VectorXf> states(trajsize);
	std::vector<float> rewardsToGo(trajsize);

	DNNGradient trajectoryPolicyGradient(net);
	float avgReward = 0;
	float gradMul = 1.0 / (float)trajectoriesPerBatch;
	Eigen::VectorXf state0(net.weights[0].cols());

	for (int i = 0; i < trajectoriesPerBatch; i++) {
		std::cout << "-- Sampling Trajectory " << i << " --" << std::endl;
		dre->sampleState(state0);
		avgReward += sampleTrajectory(net, dre, state0, actions, states, rewardsToGo, trajectoryPolicyGradient, 1.0/(float)trajectoriesPerBatch,discount,lambda);		
	}
	//print gradient
	std::cout << "gradient: " << std::endl;
	std::cout << trajectoryPolicyGradient.print();
	//apply gradient
	// negate stepSize as we are maximizing
	net.applyGradient(trajectoryPolicyGradient,-stepSize);
	std::cout << "network after update: " << std::endl;
	std::cout << net.print();
	return avgReward * gradMul;
}
float sampleTrajectoryCEM(DNN &net, DRLEnvironment* env, Eigen::VectorXf &state0) {
	std::cout << "sampling trajectory CEM... " << std::endl;
	float totalReward = 0;
	int trajsize = env->getTrajectorySize();
	std::vector<Eigen::VectorXf> outGradients(trajsize);
	std::vector<float> stepRewards(trajsize);
	Eigen::VectorXf state = state0;
	Eigen::VectorXf nextState(state0.size());
	for (int t = 0; t < trajsize; t++) {
		std::cout << "-- state " << t << ":" << state.format(RowFmt) << std::endl;
		Eigen::VectorXf policyOut = net.forward(state);
		std::cout << "-- network out: " << policyOut.format(RowFmt) << std::endl;
		Eigen::VectorXf policyGradient;
		//will fill in policy gradient for this action
		Eigen::VectorXf action(env->getActionSize());
		env->getAction(action, policyOut, policyGradient);
		std::cout << "-- action: " << action.format(RowFmt) << std::endl;
		outGradients[t] = policyGradient;
		stepRewards[t] = env->takeAction(action, state, nextState);
		std::cout << "-- reward: " << stepRewards[t] << std::endl;
		totalReward += stepRewards[t];
		state = nextState;
	}	
	return totalReward;
}
float sampleTrajectory(DNN &net, DRLEnvironment* env, Eigen::VectorXf &state0, std::vector<Eigen::VectorXf> &actions, std::vector<Eigen::VectorXf> &states, std::vector<float> &rewardsToGo, DNNGradient &gradient, float scaleFactor, float discount, float lambda) {
	int trajsize = env->getTrajectorySize();
	actions = std::vector<Eigen::VectorXf>(trajsize);
	states = std::vector<Eigen::VectorXf>(trajsize);
	rewardsToGo = std::vector<float>(trajsize,0);
	float totalReward = 0;
	std::vector<Eigen::VectorXf> outGradients(trajsize);
	std::vector<float> stepRewards(trajsize);
	Eigen::VectorXf nextState = state0;
	for(int t = 0; t < trajsize; t++){
		states[t] = nextState;
		std::cout << "-- state " << t << ":" << states[t].format(RowFmt) << std::endl;

		Eigen::VectorXf policyOut = net.forward(states[t]);
		std::cout << "-- network out: " << policyOut.format(RowFmt) << std::endl;
		Eigen::VectorXf policyGradient;
		//will fill in policy gradient for this action
		actions[t] = Eigen::VectorXf(env->getActionSize());
		env->getAction(actions[t], policyOut, policyGradient);
		std::cout << "-- action: " << actions[t].format(RowFmt) << std::endl;
		outGradients[t] = policyGradient;
		stepRewards[t] = env->takeAction(actions[t], states[t], nextState);
		std::cout << "-- reward: " << stepRewards[t] << std::endl;
		totalReward += stepRewards[t];
	}
	// rewards[t] is from action[t] that we took in step[t]
	//sum up step rewards for rewards to go and take weighted sum of gradients to get final gradient
	rewardsToGo[trajsize - 1] = stepRewards[trajsize - 1];
	StaticCost reward;

	//backprop wants ground truth out, which there isn't and our cost function doesn't need
	Eigen::VectorXf dummyAction = Eigen::VectorXf::Zero(1);
	for (int t = stepRewards.size()-1; t > -1; t--) {
		std::cout <<  " - Step " << t <<  " - " << std::endl;

		if (t > 0) {
			rewardsToGo[t - 1] = rewardsToGo[t] + stepRewards[t - 1];
		}
		//meaningless for this as it is not used for backprop
		reward.setCost(rewardsToGo[t]);
		reward.setGradient(outGradients[t]);
		std::cout << "  action: " << actions[t].format(RowFmt) << std::endl;
		std::cout << "  gradient at output: " << outGradients[t].format(RowFmt) <<  std::endl;
		//backProp(net, states[t], dummyAction, &reward, gradient, rewardsToGo[t] * scaleFactor);
		backProp(net, states[t], dummyAction, &reward, gradient, totalReward * scaleFactor);
		std::cout << "  gradient after backProp: " << std::endl;
		std::cout << gradient.print() << std::endl;
		
	}
	//by this point, actions, states, rewards to go, and gradient are all filled in
	return totalReward;
}
