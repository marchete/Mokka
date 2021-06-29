//Compile with -O3 -march=core-avx2 -mavx2 -mfma
//#define DEBUG_MODE
#ifndef _MSC_VER
#pragma GCC optimize("O3","omit-frame-pointer","inline")
//#pragma GCC option("arch=native","tune=native","no-zeroupper")
#pragma GCC target("avx2,fma")
#endif
#include <immintrin.h>
#include <chrono>
#include <ctime>
#include <ratio>
#include <iostream>
#include <iterator>
#include <memory>  //shared_ptr
#include "NN_Mokka.hpp"

#include "mnist_reader.hpp"

using namespace std;

Model CreateModel(shared_ptr<Input>& input,shared_ptr<Layer>& policy,shared_ptr<Layer>& value ){
	shared_ptr<Layer> x,split;
	#define NN(M_M) make_shared<M_M>
	input	=  NN(Input)(vector<int>{28*28});
	x		=(*NN(Dense)("Dense", 128,RELU))(input);
	policy	=(*NN(Dense)("Soft", 10,SOFTMAX))(x);
	//policy	=(*NN(Dense)("Soft", 10,NONE))(x);
	#undef NN
	Model model({input},{policy});//("simple_test28Gray.ahsf", "weights", "mean");
	return model;
}

Model CreateModel29(shared_ptr<Input>& input, shared_ptr<Layer>& policy, shared_ptr<Layer>& value) {
	shared_ptr<Layer> x, split;
#define NN(M_M) make_shared<M_M>
	input = NN(Input)(vector<int>{29 * 29});
	x = (*NN(Dense)("Dense", 127, RELU))(input);
	policy = (*NN(Dense)("Soft", 13, SOFTMAX))(x);
	//policy	=(*NN(Dense)("Soft", 10,NONE))(x);
#undef NN
	Model model({ input }, { policy });//("simple_test28Gray.ahsf", "weights", "mean");
	return model;
}
#define Now() chrono::high_resolution_clock::now()
struct Stopwatch {
	chrono::high_resolution_clock::time_point c_time, c_timeout;
	void Start(int us) { c_time = Now(); c_timeout = c_time + chrono::microseconds(us); }
	void setTimeout(int us) { c_timeout = c_time + chrono::microseconds(us); }
	inline bool Timeout() {
		return Now() > c_timeout;
	}
	long long EllapsedMicroseconds() { return chrono::duration_cast<chrono::microseconds>(Now() - c_time).count(); }
	long long EllapsedMilliseconds() { return chrono::duration_cast<chrono::milliseconds>(Now() - c_time).count(); }
} stopwatch;

/* Unfinished
Model CreateCNN(shared_ptr<Input>& input, shared_ptr<Layer>& policy, shared_ptr<Layer>& value) {
	shared_ptr<Layer> x, split;
#define NN(M_M) make_shared<M_M>
	input = NN(Input)(vector<int>{28, 28, 1});
	x = (*NN(Conv)("Conv1", 32, 3, 2, 0, RELU))(input);
	//x		=(*NN(Conv )(5, 3, 1, 0, RELU))(input);
	x = (*NN(Conv)("Conv2", 64, 3, 2, 0, RELU))(x);
	x = (*NN(Conv)("Conv3", 32, 3, 1, 0, Activation_LeakyReLU<0,100>))(x);
	x = (*NN(Dense)("Dense", 64, RELU))(x);
	policy = (*NN(Dense)("Soft", 10, SOFTMAX))(x);
#undef NN
	Model model({ input }, { policy });//("simple_test28Gray.ahsf", "weights", "mean");
	return model;
}
*/
void MNIST_inference(Model& model,int padding =0) {
	vector<Tensor> Tensor_training_images;
	vector<Tensor> Tensor_test_images;
	vector<Tensor> Tensor_training_labels;
	vector<Tensor> Tensor_test_labels;

	const bool limitedLoad = false;
	const char* MNIST_DATA_LOCATION = "./mnist/";
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
	mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION, limitedLoad ? 10 : 0, limitedLoad ? 10 : 0);
	std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
	const auto imageToTensorFn = [=](const std::vector<uint8_t>& image, Tensor& t)
	{
		//	t.initZero({ 28,28,1 });
		t = Tensor({ 1,1, (28+padding) * (28 + padding) });
		for (int y = 0; y < 28; ++y) {
			for (int x = 0; x < 28; ++x) {
				t.setElement(y *(28 + padding) + x, image[y * 28 + x] / 255.0f);
			}
		}
	};

	const auto labelToOneHot = [](uint8_t label, Tensor& t)
	{
		t = Tensor({ 1, 1, 10 });
		for (int i = 0; i < 10; i++)
		{
			t.setElement(i, i == label?1.0f:0.0f);
		}
	};

	for (auto& t : dataset.training_images)
	{
		Tensor Im;
		imageToTensorFn(t, Im);
		Tensor_training_images.push_back(Im);
	}
	for (auto& t : dataset.test_images)
	{
		Tensor Im;
		imageToTensorFn(t, Im);
		Tensor_test_images.push_back(Im);
	}
	for (auto& t : dataset.training_labels)
	{
		Tensor Im;
		labelToOneHot(t, Im);
		Tensor_training_labels.push_back(Im);
	}
	for (auto& t : dataset.test_labels)
	{
		Tensor Im;
		labelToOneHot(t, Im);
		Tensor_test_labels.push_back(Im);
	}
	int countPredict = 0;
	const int accSamplesCount = 1000;
	const auto calcAccT = [&]()
	{
		int total = 0;
		int correct = 0;
		//for (int k = 0; k < accSamplesCount; k++)
		//for (int index = 0; index < dataset.training_images.size(); ++index)
		for (int index = 0; index < Tensor_training_images.size(); ++index)
		//for (auto T: Tensor_training_images)
		{
			//const int index = g_randomGen() % Tensor_training_images.size();
			model.inputs[0]->output =  Tensor_training_images[index];
			model.predict();
			const Tensor& ans = model.outputs[0]->output;
			float MaxValue = -9999999.99f;
			uint32_t maxIndex = -1;
			for (int i = 0; i < ans.size; ++i)
			{
				if (ans.getElement(i) > MaxValue) {
					MaxValue = ans.getElement(i);
					maxIndex = i;
				}
			}
			if (dataset.training_labels[index] == maxIndex)
			{
				correct++;
			}
			total++;
			++countPredict;
		}
		return static_cast<float>(correct)*100.0 / total;
	};
	const auto calcAcc = [&]()
	{
		int total = 0;
		int correct = 0;
		//for (int k = 0; k < accSamplesCount; k++)
		for (int index = 0; index < Tensor_test_images.size(); ++index)
		{
			//const int index =  g_randomGen() % dataset.test_images.size();
			model.inputs[0]->output = Tensor_test_images[index];
			model.predict();
			const Tensor& ans = model.outputs[0]->output;
			float MaxValue = -9999999.99f;
			uint32_t maxIndex = -1;
			for (int i = 0; i < ans.size; ++i)
			{
				if (ans.getElement(i) > MaxValue) {
					MaxValue = ans.getElement(i);
					maxIndex = i;
				}
			}
			if (dataset.test_labels[index] == maxIndex)
			{
				correct++;
			}
		/*	if (index == 0)
			{
				cerr << (dataset.test_labels[index] == maxIndex ? "OK" : "MISS") << " Numero " << (int)dataset.test_labels[index] << " se ha estimado como  " << maxIndex << "(" << 100.0*ans.getElement(maxIndex) << "%) : " << ans << endl;
			}*/
			total++;
			++countPredict;
		}
		return static_cast<float>(correct)*100.0 / total;
	};


	stopwatch.Start(10);
	double f = 0.0;
	for (int i = 0; i < 10; ++i)
	{
		f+=  calcAcc() + calcAccT();
	}
	cout << " Took: " << stopwatch.EllapsedMilliseconds() << "ms mean:" << stopwatch.EllapsedMicroseconds() / countPredict << "us/sample" << endl;
	std::cout << "Est Acc:" << std::setw(7) << std::setprecision(6) << calcAcc() << "% TrainingAcc:" << std::setw(7) << std::setprecision(6) << calcAccT() << "%\n";
}


int main() {
	shared_ptr<Input> input;
	shared_ptr<Layer> policy,value;

	cerr << "*****************************************************"<<endl;
	cerr << "********** TEST 'MNIST Simple.ipynb' ****************"<<endl;
	cerr << "********** Using dense.weights       ****************"<<endl;
	cerr << "*****************************************************"<<endl;
	Model model = CreateModel(input, policy, value);
	model.summary();
	model.loadWeights("DENSE.weights");
	model.saveWeights("DENSE.test");
	MNIST_inference(model);
	

	cerr << "*****************************************************"<<endl;
	cerr << "********** TEST 'MNIST Simple29.ipynb' **************"<<endl;
	cerr << "********** Using dense29.weights       **************"<<endl;
	cerr << "*****************************************************"<<endl;

	Model model29 = CreateModel29(input, policy, value);
	model29.summary();
	model29.loadWeights("DENSE29.weights");
	model29.saveWeights("DENSE29.test");
	MNIST_inference(model29,1);

/*
	Model CNNmodel = CreateCNN(input, policy, value);
	CNNmodel.summary();
	CNNmodel.loadWeights("PRUEBA.weights");
	CNNmodel.saveWeights("PRUEBA.test");
	MNIST_inference(CNNmodel);
	*/



    return 0;
}