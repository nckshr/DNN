#include "lib/Eigen/Dense"
#include "DNN.h"
#include "DRL.h"
#include "Sampling.h"
#include <stdio.h>
#include <iostream>
void DNNTest() {
	/*
// SHOWS VECTORS EXTRACTED FROM MATRIX ROWS ARE STILL COLUMN VECTORS

Eigen::MatrixXf x = Eigen::MatrixXf::Random(10, 5);
Eigen::VectorXf y1 = x.row(0);
Eigen::VectorXf y2 = x.col(0);
std::cout << "row is of size: " << y1.rows() << ", " << y1.cols() << std::endl;
std::cout << "col is of size: " << y2.rows() << ", " << y2.cols() << std::endl;
*/
	DNN net = DNN();
	//input layer
	net.addLayer(2);
	//hidden layer
	net.addLayer(5);
	//output layer
	net.addLayer(2);
	Eigen::MatrixXf data = (Eigen::MatrixXf::Random(1000, 2));
	/*
	Eigen::MatrixXf data(10, 2);
	data << 1, 0,
		0, 1,
		-1, 0,
		0, -1,
		0, 1,
		1, 0,
		0, -1,
		-1, 0,
		0.5, 0.5,
		-0.5, 0.5;
		*/
	MSE mse = MSE();
	MAE mae = MAE();

	train(net, data, data, 50, 10, 0.005, 0.95, &mae, "sgd");

	std::cout << "resulting network:" << std::endl;
	std::cout << net.print();
	std::cout << "saving network... ";
	net.saveModel("identityModel");
	std::cout << "done. \n Reading back in...";
	net.loadModel("identityModel");
	std::cout << "done. \n Loaded Network:\n";
	std::cout << net.print();

	std::cout << "Testing on some random rows: " << std::endl;
	for (int i = 0; i < 5; i++) {
		Eigen::VectorXf x = data.row(i);
		Eigen::VectorXf y = data.row(i);
		Eigen::VectorXf yhat = net.forward(x);
		Eigen::VectorXf gradient(yhat.size());
		std::cout << " ---- TEST " << std::setw(3) << i << " ----" << std::endl;
		std::cout << "  input: \n" << x.format(CleanFmt) << std::endl;
		std::cout << "  ground truth: \n" << y.format(CleanFmt) << std::endl;
		std::cout << "  network output : \n" << yhat.format(CleanFmt) << std::endl;
		std::cout << "  cost: " << mae.getCost(y, yhat, gradient) << std::endl;
	}
}
void testGaussianSampling() {
	Eigen::VectorXf samples(100);
	for (int i = 0; i < samples.size(); i++) {
		samples[i] = sampleGaussian(0, 1);
	}
	std::cout << samples.format(ListFmt);
}
void DRLActionSampleTest() {
	DNN policy = DNN();
	std::cout << "initializing policy..." << std::endl;
	//input a2, a2, a3, x, y, z, gx, gy, gz
	policy.addLayer(9);
	policy.addLayer(20);
	policy.addLayer(15);
	policy.addLayer(10);
	policy.addLayer(6);

	Eigen::VectorXf state(9);
	state << 6.281, 0.728 , 6.109 , -0.7149 , 1.758 , -0.001271 , 0.2126 , 1.127 , -0.8976;
	Eigen::VectorXf policyOut = policy.forward(state);
	Eigen::VectorXf action(3);
	Eigen::VectorXf policyGradient = Eigen::VectorXf(6);
	for (int i = 0; i < 3; i++) {
		float sigma = std::exp(policyOut[i + 3]);
		action[i] = sampleGaussian(policyOut[i], sigma);
		policyGradient[i] = (action[i] - policyOut[i]) / std::powf(sigma, 2.0);
		policyGradient[i + 3] = policyGradient[i] * (action[i] - policyOut[i]);
	}
	std::cout << "action from policy: " << action.format(CleanFmt) << std::endl;
}
void CEMTest() {
	DNN net = DNN();
	//input layer
	int dim = 2;
	net.addLayer(dim);
	//hidden layer
	//net.addLayer(5);
	//output layer
	net.addLayer(dim);
	Eigen::MatrixXf data = (Eigen::MatrixXf::Random(1000, dim));
	MSE mse = MSE();
	trainCEM(net, data, data, 100, 1000, 1000, 100, &mse);
	std::cout << "resulting network:" << std::endl;
	std::cout << net.print();
	std::cout << "saving network... ";
	net.saveModel("identityModel");
	std::cout << "done. \n Reading back in...";
	net.loadModel("identityModel");
	std::cout << "done. \n Loaded Network:\n";
	std::cout << net.print();
	std::cout << "Testing on some random rows: " << std::endl;
	for (int i = 0; i < 5; i++) {
		Eigen::VectorXf x = data.row(i);
		Eigen::VectorXf y = data.row(i);
		Eigen::VectorXf yhat = net.forward(x);
		Eigen::VectorXf gradient(yhat.size());
		std::cout << " ---- TEST " << std::setw(3) << i << " ----" << std::endl;
		std::cout << "  input: \n" << x.format(CleanFmt) << std::endl;
		std::cout << "  ground truth: \n" << y.format(CleanFmt) << std::endl;
		std::cout << "  network output : \n" << yhat.format(CleanFmt) << std::endl;
		std::cout << "  cost: " << mse.getCost(y, yhat, gradient) << std::endl;
	}
}
void matrixIndexTest() {
	Eigen::MatrixXf m(3, 2);
	m << 1, 2, 3, 4, 5, 6;
	std::cout << "matrix before: " << m.format(CleanFmt);
	float* data = m.data();
	for (int i = 0; i < 6; i++) {
		data[i] = 0;
	}
	std::cout << "matrix after: " << m.format(CleanFmt);
}
void sortTest() {
	std::vector<float> x = { 5,2,4,3,1,7,9 };
	std::vector<intfloatPair> sorted = sortWithIndexReturn(x);
	for (int i = 0; i < x.size(); i++) {
		std::cout << sorted[i].second << ", ";
	}
	std::cout << std::endl;
}
void learnMNIST() {
	DNN net = DNN();
	std::cout << "loading data..." << std::endl;
	Eigen::MatrixXf testDataRaw = matrixFromFile("E:\\Users\\ncksh\\Code\\mnist_data\\mnist_test.csv",0,',');
	std::cout << "loaded " << testDataRaw.rows() << " x " << testDataRaw.cols() << " test file" << std::endl;
	Eigen::MatrixXf trainDataRaw = matrixFromFile("E:\\Users\\ncksh\\Code\\mnist_data\\mnist_train.csv", 0, ',');
	std::cout << "loaded " << trainDataRaw.rows() << " x " << trainDataRaw.cols() << " training file" << std::endl;
	std::cout << "done." << std::endl;
	//normalize pixel values
	std::cout << "done." << std::endl;	
	Eigen::VectorXf testClasses = testDataRaw.col(0);
	Eigen::VectorXf trainClasses = trainDataRaw.col(0);
	std::cout << "extracted class vectors..." << std::endl;
	trainDataRaw = trainDataRaw * (1.0 / 255.0);
	testDataRaw = testDataRaw * (1.0 / 255.0);
	std::cout << "normalized pixel values..." << std::endl;
	int testDataRows = testDataRaw.rows();
	int testDataCols = testDataRaw.cols() - 1;
	int trainDataRows = trainDataRaw.rows();
	int trainDataCols = trainDataRaw.cols() - 1;

	Eigen::MatrixXf testData(testDataRows, testDataCols);
	Eigen::MatrixXf trainData(trainDataRows, trainDataCols);
	
	testData.block(0,0, testDataRows, testDataCols) = testDataRaw.block(0, 1, testDataRows, testDataCols);
	trainData.block(0, 0, trainDataRows, trainDataCols) = trainDataRaw.block(0, 1, trainDataRows, trainDataCols);
	std::cout << "removed class vectors from data..." << std::endl;

	//one-hot encoded
	Eigen::MatrixXf testY = Eigen::MatrixXf::Zero(testData.rows(),10);
	Eigen::MatrixXf trainY = Eigen::MatrixXf::Zero(trainData.rows(),10);

	for (int i = 0; i < trainData.rows(); i++) {
		trainY(i,((int)trainClasses[i])) = 1.0;
	}
	for (int i = 0; i < testData.rows(); i++) {
		testY(i, ((int)testClasses[i])) = 1.0;
	}
	std::cout << "one-hot encoded class vectors..." << std::endl;

	//print out some lables
	/*
	std::cout << "first 5 labels: " << std::endl;
	for (int i = 0; i < 5; i++) {
		std::cout << trainY.row(i).format(CleanFmt) << std::endl;
	}
	std::cout << "first 5 rows: " << std::endl;
	for (int i = 0; i < 5; i++) {
		std::cout << trainData.row(i).format(CleanFmt) << std::endl;
	}
	*/
	CatCrossEntropy crossEntropy = CatCrossEntropy();
	int iSize = testDataRaw.cols() - 1;
	net.addLayer(iSize);
	net.addLayer(800);
	net.addLayer(10);
	std::cout << "training..." << std::endl;
	train(net, trainData, trainY, testData, testY, 3,100, 0.001, 1, &crossEntropy, "sgd");

	std::cout << "saving network... ";
	net.saveModel("mnistModel");
	//std::cout << "done. \n Reading back in...";
	//net.loadModel("mnist");
	//std::cout << "done. \n Loaded Network:\n";
	//std::cout << net.print();
	//std::cout << "Testing on some random rows: " << std::endl;
	std::cout << "Testing data: " << std::endl;
	float acc = 0;
	for (int i = 0; i < testData.rows(); i++) {
		Eigen::VectorXf x = testData.row(i);
		Eigen::VectorXf y = testY.row(i);
		Eigen::VectorXf yhat = net.forward(x);
		Eigen::VectorXf gradient(yhat.size());
		if (argmax(yhat) == argmax(y)) {
			acc += 1.0;
		}
		/*
		if (i % 10 == 0) {
			std::cout << " ---- TEST " << std::setw(3) << i << " ----" << std::endl;
			//std::cout << "  input: \n" << x.format(CleanFmt) << std::endl;
			std::cout << "  ground truth: \n" << y.format(CleanFmt) << std::endl;
			std::cout << "  network output : \n" << yhat.format(CleanFmt) << std::endl;
			std::cout << "  cost: " << crossEntropy.getCost(y, yhat, gradient) << std::endl;
		}
		*/
	}
	std::cout << "TEST ACCURACY: " << acc / ((float)testData.rows()) << "%" << std::endl;
}
int main(int argc, char* argv[]) {
	//testGaussianSampling();
	//DRLActionSampleTest();
	//matrixIndexTest();
	//CEMTest();
	learnMNIST();
	return EXIT_SUCCESS;
}
