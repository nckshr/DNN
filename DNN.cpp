#include "DNN.h"
#include <filesystem>
DNN::DNN(){
	inputSize = -1;
	weights = std::vector<Eigen::MatrixXf>();
	biases = std::vector<Eigen::VectorXf>();
	prelus = std::vector<Eigen::VectorXf>();
}
DNN::DNN(const char* modelFolder) {
	readModel(modelFolder, weights, biases, prelus);
}
std::string DNN::print() {
	std::stringstream outStr("");
	for (int w = 0; w < weights.size(); w++) {
		outStr << "--- LAYER " << w << "---" << std::endl;
		outStr << "W" << w << ":" << std::endl;
		outStr << weights[w].format(CleanFmt) << std::endl;
		outStr << "b" << w << ":" << std::endl;
		outStr << biases[w].format(CleanFmt) << std::endl;
	}
	return outStr.str();
}
void DNN::init() {
	inputSize = -1;
	weights = std::vector<Eigen::MatrixXf>();
	biases = std::vector<Eigen::VectorXf>();
	prelus = std::vector<Eigen::VectorXf>();
}
void DNN::addLayer(int size) {
	if (inputSize < 0) {
		//this is the input
		inputSize = size;
	}
	else {
		//add a weight matrix, bias, and prelu to the output of this layer
		//initialize to small, positive values
		int prevLayerSize = weights.size() == 0 ? inputSize : weights[weights.size() - 1].rows();
		//He et al. 2015
		float scale = std::sqrtf(2.0f / (float)prevLayerSize);
		weights.push_back(Eigen::MatrixXf::Random(size, prevLayerSize).array() * scale);
		biases.push_back(Eigen::VectorXf::Random(size).array() * scale);
		//prelu will be ignored in feed forward for last layer as it is the output layer
		prelus.push_back(Eigen::VectorXf::Zero(size));
	}
}
void DNN::loadModel(const char* modelFolder) {
	readModel(modelFolder, weights, biases, prelus);
}
Eigen::VectorXf DNN::forward(Eigen::VectorXf input) {
	return feedForward(input, weights, biases, prelus);
}
void DNN::applyGradient(DNNGradient &gradient, float stepSize) {
	for (int w = 0; w < weights.size(); w++) {		
		//std::cout << "weight gradient at layer " << w << ": " << std::endl;
		//std::cout << weightGradients[w].format(CleanFmt) << std::endl;
		weights[w] = weights[w] - stepSize * gradient.dCost_dWeights[w];
		//std::cout << "updated weights at layer " << w << ": " << std::endl;
		//std::cout << net.weights[w].format(CleanFmt) << std::endl;
		//std::cout << "bias gradient at layer " << w << ": " << std::endl;
		//std::cout << biasGradients[w].format(CleanFmt)<<std::endl;
		biases[w] = biases[w] - stepSize * gradient.dCost_dBiases[w];
		//std::cout << "updated biases at layer " << w << ": " << std::endl;
		//std::cout << net.biases[w].format(CleanFmt) << std::endl;

	}
}
void readModel(const char* modelFolder, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases) {
	std::vector<Eigen::VectorXf> dummyPrelus(0);
	readModel(modelFolder, weights, biases, dummyPrelus);
}
void DNN::saveModel(const char* modelFolder) {
	if (!std::filesystem::exists(modelFolder)) {
		std::filesystem::create_directory(modelFolder);
	}
	else if (!std::filesystem::is_directory(modelFolder)) {
		std::filesystem::create_directory(modelFolder);
	}
	for (int layer = 0; layer < weights.size(); layer++) {
		std::stringstream weightFile("");
		std::stringstream biasFile("");
		std::stringstream preluFile("");
		weightFile << modelFolder << "/" << "w" << layer << ".csv";
		biasFile << modelFolder << "/" << "b" << layer << ".csv";
		preluFile << modelFolder << "/" << "p" << layer << ".csv";
		matrixToFile(weightFile.str().c_str(), weights[layer]);
		vectorToFile(biasFile.str().c_str(), biases[layer]);
		vectorToFile(preluFile.str().c_str(), prelus[layer]);
	}
}
void readModel(const char* modelFolder, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus) {
	std::list<Eigen::MatrixXf> weightList;
	std::list<Eigen::VectorXf> biasList;
	std::list<Eigen::VectorXf> preluList;
	int layer = 0;
	bool done = false;
	errno_t werr = 0;
	errno_t berr = 0;
	errno_t perr = 0;
	FILE* filepointer;
	while (!done) {
		std::stringstream weightFile("");
		std::stringstream biasFile("");
		std::stringstream preluFile("");

		weightFile << modelFolder << "/" << "w" << layer << ".csv";
		biasFile << modelFolder << "/" << "b" << layer << ".csv";
		preluFile << modelFolder << "/" << "p" << layer << ".csv";
		werr = fopen_s(&filepointer, weightFile.str().c_str(), "r");
		if (werr == 0) {
			fclose(filepointer);
		}
		else {
			done = true;
			break;
		}
		berr = fopen_s(&filepointer, biasFile.str().c_str(), "r");
		if (berr == 0) {
			fclose(filepointer);
		}
		perr = fopen_s(&filepointer, preluFile.str().c_str(), "r");
		if (perr == 0) {
			fclose(filepointer);
		}
	
		if (werr == 0) {
			weightList.push_back(matrixFromFile(weightFile.str().c_str(), 0));
		}
		else {
			done = true; 
			break;
		}
		if (berr == 0) {
			biasList.push_back(vectorFromFile(biasFile.str().c_str(),false));
		}
		else {
			biasList.push_back(Eigen::VectorXf::Zero(weightList.back().rows()));
		}
		if (perr == 0) {
			preluList.push_back(vectorFromFile(preluFile.str().c_str(),false));
		}
		else {
			preluList.push_back(Eigen::VectorXf::Zero(weightList.back().rows()));
		}
		layer++;
	}
	std::cout << "Initialized network with " << layer << " layers: Output ";
	weights = std::vector<Eigen::MatrixXf>{ std::begin(weightList), std::end(weightList) };
	biases = std::vector<Eigen::VectorXf>{ std::begin(biasList), std::end(biasList) };
	prelus = std::vector<Eigen::VectorXf>{ std::begin(preluList),std::end(preluList) };
	for (int w = weights.size() - 1; w > -1; w--) {
		std::cout << "<=(" << weights[w].rows() << "x" << weights[w].cols() << ")";
	}
	std::cout << " <= Input " << std::endl;
}
void train(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int numEpochs, int batchSize, float learningRate, float learningDecay, CostFunction* costFunc, std::string method) {
	srand(time(NULL));
	float step = learningRate;
	for (int i = 0; i < numEpochs; i++) {
		float avgAvgCost = runEpoch(net, trainIn, trainGroundTruth, batchSize, step, costFunc);
		std::cout << "\n -- EPOCH " << std::setfill(' ') << std::setw(5) << i << " / " << numEpochs << " -- \n Average Batch Mean Cost: " << std::setfill(' ') << std::setw(10) << avgAvgCost << std::endl;
		step = step * learningDecay;
	}
}
void train(DNN &net, Eigen::MatrixXf &trainX, Eigen::MatrixXf &trainY, Eigen::MatrixXf &testX, Eigen::MatrixXf &testY, int numEpochs, int batchSize, float learningRate, float learningDecay, CostFunction* costFunc, std::string method) {
	srand(time(NULL));
	float step = learningRate;
	for (int i = 0; i < numEpochs; i++) {
		float avgAvgCost = runEpoch(net, trainX, trainY, batchSize, step, costFunc);
		float testCost = 0;
		//compute avg test loss
		for (int j = 0; j < testX.rows(); j++) {
			Eigen::VectorXf yhat = net.forward(testX.row(j));
			Eigen::VectorXf gradient;
			Eigen::VectorXf y = testY.row(j);
			testCost += costFunc->getCost(y, yhat, gradient);
		}
		testCost = testCost / (float)testX.rows();
		std::cout << "\n -- EPOCH " << std::setfill(' ') << std::setw(5) << i << " / " << numEpochs << " -- \n Average Batch Mean Cost: " << std::setfill(' ') << std::setw(10) << avgAvgCost << "\n Average Test Cost: " << std::setfill(' ') << std::setw(10) << testCost << std::endl;
		step = step * learningDecay;
	}
}
//at the epoch level, do things like adjust learning rate and shuffle batch order
float runEpoch(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int batchSize, float stepSize, CostFunction* costFunc) {
	float avgAvgCost = 0;
	int numBatches = trainIn.rows() / batchSize;
	int lastBatchSize = trainIn.rows() % batchSize;
	if (lastBatchSize > 0) {
		numBatches++;
	}
	else {
		lastBatchSize = batchSize;
	}
	std::vector<float> batchOrder(numBatches);
	for(int i = 0; i < batchOrder.size(); i++){
		batchOrder[i] = rand();
	}
	std::vector<intfloatPair> sortedBatchNums = sortWithIndexReturn(batchOrder);

	int offset = 0;
	for (int n = 0; n < numBatches; n++) {
		int batchNum = sortedBatchNums[n].first;
		offset = batchNum * batchSize;
		int thisBatchSize = batchSize;
		if (batchNum == numBatches-1) {
			int thisBatchSize = lastBatchSize;
		}		
		float avgCost = runBatch(net,trainIn,trainGroundTruth,costFunc, thisBatchSize,offset,stepSize);
		std::cout << "\r batch " << std::setfill(' ') << std::setw(5) << n << " / " << numBatches << " (#"<< batchNum <<std::setfill(' ') << std::setw(5) <<") : " << std::setfill(' ') << std::setw(10) << avgCost << std::flush;
		avgAvgCost += avgCost;
	}
	return avgAvgCost/(float)numBatches;
}
// at the batch level, sum gradients over the data and return the average as the gradient to be applied
float runBatch(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, CostFunction* costFunc, int batchSize, int offset, float stepSize) {
	DNNGradient gradient(net);
	float gradMul = 1.0 / ((float)batchSize);
	
	float totalCost = 0;
	for (int i = offset; i < offset+ batchSize; i++) {
		
		Eigen::VectorXf input = trainIn.row(i);
		Eigen::VectorXf groundTruth = trainGroundTruth.row(i);
		//std::cout << "input: \n" << input.format(CleanFmt) << std::endl;
		//std::cout << "ground truth: \n" << groundTruth.format(CleanFmt) << std::endl;
		totalCost += backProp(net, input, groundTruth, costFunc, gradient, gradMul);
	}
	//now gradients holds the averaged gradient for each weight matrix. Apply according to given stepSize
	net.applyGradient(gradient, stepSize);
	//return average cost over the batch
	return totalCost * gradMul;
}
void trainCEM(DNN &net, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int numEpochs, int batchSize, int poolSize, int winners, CostFunction* costFunc) {
	srand(time(NULL));
	std::vector<DNN> samples(poolSize);
	makeCEMSamples(net, samples, poolSize);
	for (int i = 0; i < numEpochs; i++) {
		float avgAvgCost = runEpochCEM(samples, trainIn, trainGroundTruth, batchSize, winners, costFunc);
		std::cout << "\n -- EPOCH " << std::setfill(' ') << std::setw(5) << i << " / " << numEpochs << " -- \n Average Batch Mean Cost: " << std::setfill(' ') << std::setw(10) << avgAvgCost << std::endl;
	}
	//grab a sample as the resulting network
	net = samples[0];
}
float runEpochCEM(std::vector<DNN> &samples, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, int batchSize, int winners, CostFunction* costFunc) {
	float avgAvgCost = 0;
	int numBatches = trainIn.rows() / batchSize;
	int lastBatchSize = trainIn.rows() % batchSize;
	if (lastBatchSize > 0) {
		numBatches++;
	}
	else {
		lastBatchSize = batchSize;
	}
	std::vector<float> batchOrder(numBatches);
	for (int i = 0; i < batchOrder.size(); i++) {
		batchOrder[i] = rand();
	}
	std::vector<intfloatPair> sortedBatchNums = sortWithIndexReturn(batchOrder);

	int offset = 0;
	for (int n = 0; n < numBatches; n++) {
		int batchNum = sortedBatchNums[n].first;
		offset = batchNum * batchSize;
		int thisBatchSize = batchSize;
		if (batchNum == numBatches - 1) {
			int thisBatchSize = lastBatchSize;
		}
		float avgCost = runBatchCEM(samples, trainIn, trainGroundTruth, costFunc, thisBatchSize, offset, winners);
		std::cout << "\r batch " << std::setfill(' ') << std::setw(5) << n << " / " << numBatches << " (#" << batchNum << std::setfill(' ') << std::setw(5) << ") : " << std::setfill(' ') << std::setw(10) << avgCost << std::flush;
		avgAvgCost += avgCost;
	}
	return avgAvgCost / (float)numBatches;
}
float runBatchCEM(std::vector<DNN> &samples, Eigen::MatrixXf &trainIn, Eigen::MatrixXf &trainGroundTruth, CostFunction* costFunc, int batchSize, int offset, int winners) {
	//std::cout << "evaluating samples... " << std::endl;
	std::vector<float> avgCosts(samples.size(), 0);
	int nlayers = samples[0].weights.size();
	float scale = (1.0 / (float)batchSize);
	for (int a = 0; a < samples.size(); a++) {
		for (int i = offset; i < offset + batchSize; i++) {
			Eigen::VectorXf x = trainIn.row(i);
			Eigen::VectorXf y = trainGroundTruth.row(i);
			Eigen::VectorXf yhat = samples[a].forward(x);
			Eigen::VectorXf dummyGradient;
			avgCosts[a] += costFunc->getCost(y, yhat, dummyGradient) * scale;
		}
	}
	//std::cout << "building distributions... " << std::endl;
	//now we have average sample costs on the same data. Collect best of the bunch and resample new set of samples
	std::vector<std::vector<float>>weightMeans(nlayers);
	std::vector<std::vector<float>>biasMeans(nlayers);
	std::vector<std::vector<float>>weightVars(nlayers);
	std::vector<std::vector<float>>biasVars(nlayers);
	for (int l = 0; l < nlayers; l++) {
		int rows = samples[0].weights[l].rows();
		int cols = samples[0].weights[l].cols();
		int b = samples[0].biases[l].size();
		weightMeans[l] = std::vector<float>(rows*cols, 0);
		weightVars[l] = std::vector<float>(rows*cols, 0);
		biasMeans[l] = std::vector<float>(b, 0);
		biasVars[l] = std::vector<float>(b, 0);
	}
	//std::cout << "collecting best agents... " << std::endl;
	float totalCosts = 0;
	//TODO: Replace this with min-heap for speed
	//you really want to do min-heap for this...
	//will sort increasing
	std::vector<intfloatPair> sortedCosts = sortWithIndexReturn(avgCosts);
	//take the top x % of networks and update parameter gaussians
	//compute sample means & stdevs
	//n-1 for sample distribution (because stats)
	//std::cout << "building sample distributions... " << std::endl;
	float wscale = (1.0f / (float)(winners - 1));
	//std::cout << "winners: " << std ::endl;
	for (int i = 0; i < winners; i++) {
		int ix = sortedCosts[i].first;
		totalCosts += sortedCosts[i].second;
		//std::cout << "-- winner -- " << i << " (score: " << sortedCosts[i].second << "): " << std::endl;
		//std::cout << samples[ix].print() << std::endl;
		for (int l = 0; l < samples[ix].weights.size(); l++) {
			int rows = samples[ix].weights[l].rows();
			int cols = samples[ix].weights[l].cols();
			float* wdata = samples[ix].weights[l].data();
			for (int w = 0; w < rows*cols; w++) {
				weightMeans[l][w] += wdata[w] * wscale;
			}
			for (int b = 0; b < rows; b++) {
				biasMeans[l][b] += samples[ix].biases[l][b] * wscale;
			}

		}
		for (int l = 0; l < samples[ix].weights.size(); l++) {
			int rows = samples[ix].weights[l].rows();
			int cols = samples[ix].weights[l].cols();
			float* wdata = samples[ix].weights[l].data();

			for (int w = 0; w < rows*cols; w++) {
				weightVars[l][w] += std::powf(wdata[w] - weightMeans[l][w], 2.0) * wscale;
			}
			for (int b = 0; b < rows; b++) {
				biasVars[l][b] += std::powf(samples[ix].biases[l][b] - biasMeans[l][b], 2.0) * wscale;
			}

		}
	}
	/*
	std::cout << "distributions:" << std::endl;
	for (int l = 0; l < nlayers; l++) {
		std::cout << "-- Layer " << l << " --" << std::endl;
		std::cout << " w: ";
		for (int w = 0; w < weightMeans[l].size(); w++) {
			std::cout << weightMeans[l][w] << " +/- " << std::powf(weightVars[l][w], 0.5f) << ", ";
		}
		std::cout << std::endl;
		std::cout << " b: ";
		for (int b = 0; b < biasMeans[l].size(); b++) {
			std::cout << biasMeans[l][b] << " +/- " << std::powf(biasVars[l][b], 0.5f) << ", ";
		}
		std::cout << std::endl;
	}
	*/
	//resample
	for (int i = 0; i < samples.size(); i++) {
		for (int l = 0; l < samples[i].weights.size(); l++) {
			
			int rows = samples[i].weights[l].rows();
			int cols = samples[i].weights[l].cols();
			float* wdata = samples[i].weights[l].data();
			for (int w = 0; w < rows*cols; w++) {
				wdata[w] = sampleGaussian(weightMeans[l][w], std::powf(weightVars[l][w], 0.5f));
			}
			for (int b = 0; b < rows; b++) {
				samples[i].biases[l][b] = sampleGaussian(biasMeans[l][b], std::powf(biasVars[l][b], 0.5f));
			}
		}
	}
	return totalCosts / (float)winners;
}
void makeCEMSamples(DNN &model, std::vector<DNN> &agents, int poolSize) {
	std::cout << "making CEM agents... ";
	for (int i = 0; i < poolSize; i++) {
		DNN policy = DNN();
		//use output learner as template
		policy.addLayer(model.inputSize);
		for (int l = 0; l < model.weights.size(); l++) {
			float scale = l == model.weights.size() - 1 ? 0.1f : 1.0f;
			policy.addLayer(model.weights[l].rows());

			int rows = policy.weights[l].rows();
			int cols = policy.weights[l].cols();
			float* wdata = policy.weights[l].data();
			for (int w = 0; w < rows*cols; w++) {
				wdata[w] = sampleGaussian(0.0f, 1.0f) * scale;
			}
			for (int b = 0; b < rows; b++) {
				policy.biases[l][b] = sampleGaussian(0.0f, 1.0f) * scale;
			}

		}
		agents[i] = policy;
	}
}
// for a single input-ground_truth pair, compute the weight gradients
float backProp(DNN &net, Eigen::VectorXf &input, Eigen::VectorXf &groundTruthOut, CostFunction* costFunc, DNNGradient &gradient, float gradientMultiplier) {
	std::vector<Eigen::VectorXf> layerVals(net.weights.size());
	Eigen::VectorXf dLayerOut;
	Eigen::VectorXf yhat = feedForward(input, net.weights,net.biases,net.prelus,layerVals);
	//std::cout << "yhat : \n"<< yhat.format(CleanFmt) << std::endl;
	float cost = costFunc->getCost(groundTruthOut, yhat, dLayerOut);
	for (int l = net.weights.size()-1; l > -1; l--) {
		//layerVals are stored before the relu, need to apply it
		//layerVals will include an empty output layer, adjust index
		Eigen::VectorXf layerInAfterPrelu;
		Eigen::VectorXf layerIn;
		if (l == 0) {
			layerIn = input;
			layerInAfterPrelu = input;
		}
		else {
			layerIn = layerVals[l - 1];
			layerInAfterPrelu = layerVals[l - 1];
			applyPrelu(net.prelus[l - 1], layerInAfterPrelu);
		}
		gradient.dCost_dBiases[l] = gradient.dCost_dBiases[l] + dLayerOut * gradientMultiplier;
		gradient.dCost_dWeights[l] = gradient.dCost_dWeights[l] + dLayerOut * layerInAfterPrelu.transpose() * gradientMultiplier;		
		dLayerOut = net.weights[l].transpose() * dLayerOut;
		//pass dLayerOut backwards throug the prelu based on layerIn values
		if (l > 0) {
			for (int p = 0; p < net.prelus[l - 1].size(); p++) {
				dLayerOut[p] = layerIn[p] < 0 ? net.prelus[l - 1][p] * dLayerOut[p] : dLayerOut[p];
			}
		}
		//std::cout << "dLayerOut at layer " << l << ": \n" << std::endl << dLayerOut.format(CleanFmt) << std::endl;
	}
	return cost;
}
void applyPrelu(Eigen::VectorXf &prelu, Eigen::VectorXf &vals) {
	for (int j = 0; j < vals.size(); j++) {		
		//PRELU
		//std::cout << "applying prelu at layer " << i << std::endl;
		//std::cout << prelus[i].size() << std::endl;
		vals[j] = vals[j] < 0 ? prelu[j] * vals[j] : vals[j];
	}
}

Eigen::VectorXf feedForward(Eigen::VectorXf &inputs, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus) {
	std::vector<Eigen::VectorXf> intermediateValues = std::vector<Eigen::VectorXf>(0);
	Eigen::VectorXf output(weights[weights.size() - 1].rows());
	feedForward(inputs, output, weights, biases,prelus, 0, weights.size(), intermediateValues);
	return output;
}
Eigen::VectorXf feedForward(Eigen::VectorXf &inputs, std::vector<Eigen::MatrixXf> &weights, std::vector<Eigen::VectorXf> &biases, std::vector<Eigen::VectorXf> &prelus, std::vector<Eigen::VectorXf> &intermediateValues) {
	Eigen::VectorXf output(weights[weights.size() - 1].rows());
	feedForward(inputs, output, weights, biases, prelus, 0, weights.size(), intermediateValues);
	return output;
}

void feedForward(Eigen::VectorXf &inputs, 
	Eigen::VectorXf &output, 
	std::vector<Eigen::MatrixXf> &weights, 
	std::vector<Eigen::VectorXf> &biases, 
	std::vector<Eigen::VectorXf> &prelus, 
	int fromLayer, int toLayer, 
	std::vector<Eigen::VectorXf> &intermediateValues) {
	//output is class id
	//inputs are assumed normalized where appropriate
	//convert to row vector
	Eigen::VectorXf layerin = inputs;
	//std::cout << "input size: " << layerin.rows() << " x " << layerin.cols() << std::endl;
	for (int i = fromLayer; i < toLayer; i++) {
		if (weights[i].cols() != layerin.rows()) {
			std::cout << "Input and weights have incompatible dimensions at layer " << i << "!!";
			std::cout << "Input size: " << layerin.rows() << ", expected size: " << weights[i].cols() << std::endl;
			std::exit(1);
		}
		//layerin = weights[i].transpose() * layerin;
		layerin = weights[i] * layerin;
		//std::cout << "layerin size after applying " << weights[i].rows() << "x " << weights[i].cols() << " weight matrix: " << layerin.size() << std::endl;
		layerin = layerin +biases[i];
		//std::cout << "layerin size after applying " << biases[i].rows() << "x " << weights[i].cols() << " bias: " << layerin.size() << std::endl;
		
		if (i < intermediateValues.size()) {
			intermediateValues[i] = Eigen::VectorXf(layerin.size());
		}
		
		//component wise PRELU
		if (i < weights.size() - 1) {
			for (int j = 0; j < layerin.size(); j++) {
				if(i < intermediateValues.size()) intermediateValues[i][j] = layerin[j];
				//RELU
				//layerin[j] = std::max(0.0, layerin[j]);

				//PRELU
				//std::cout << "applying prelu at layer " << i << std::endl;
				//std::cout << prelus[i].size() << std::endl;
				layerin[j] = layerin[j] < 0 ? prelus[i][j] * layerin[j]: layerin[j];
			}
		}
	}
	for (int i = 0; i < layerin.size(); i++) {
		output[i] = layerin[i];
	}
}
