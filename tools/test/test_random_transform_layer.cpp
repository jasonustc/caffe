#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/random_layers.hpp"


float sigmoid_test(float input){
	return 1. / (1. + exp(-input));
}

float tanh_test(float input){
	return 2. * sigmoid_test(2. * input) - 1.;
}

namespace caffe{
	template <typename Dtype>
	class RandomTransformTest{
	public:
		RandomTransformTest() : x_(new Blob<Dtype>()), x_trans_(new Blob<Dtype>()){
			this->SetUp();
		}

		~RandomTransformTest(){  delete x_; delete x_trans_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 4);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 2);
			CHECK_EQ(top_[0]->shape(3), 3);
		}

		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
		}

		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
		}

		void TestCPUGradients(){
			RandomTransformLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			//checker.CheckGradientExhaustive(&layer, bottom_, top_);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			LOG(INFO) << top_[0]->count();
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
		}

		void TestGPUGradients(){
			RandomTransformLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
		}
		
	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(4);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_shape.push_back(3);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			bottom_.push_back(x_);
			top_.push_back(x_trans_);
			propagate_down_.resize(1, true);

			//set layer param
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* x_trans_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	caffe::RandomTransformTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUGradients();
	test.TestGPUForward();
	test.TestGPUGradients();
	return 0;
}