#include <vector>
#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/common_layers.hpp"

namespace caffe{
	template <typename Dtype>
	class AdaptiveDropoutTest{
	public: 
		AdaptiveDropoutTest() : bottom_(new Blob<Dtype>()), top_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<AdaptiveDropoutLayer<Dtype>> layer(new AdaptiveDropoutLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_vec_, top_vec_);
			LOG(INFO) << top_vec_[0]->shape_string();
			CHECK_EQ(top_vec_[0]->shape(0), 2);
			CHECK_EQ(top_vec_[0]->count(1), 3);
		}
		void TestCPUForward(){
			shared_ptr<AdaptiveDropoutLayer<Dtype>> layer(new AdaptiveDropoutLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_vec_, top_vec_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_vec_, top_vec_);
			top_vec_[0]->ToTxt("top_data");
		}

		void TestGPUForward(){
			AdaptiveDropoutLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_vec_, top_vec_);
			layer.Forward(bottom_vec_, top_vec_);
			top_vec_[0]->ToTxt("top_data");
		}

		void TestCPUGradients(){
			AdaptiveDropoutLayer<Dtype> layer(layer_param_);
			layer.SetUp(bottom_vec_, top_vec_);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_vec_, top_vec_);
		}

		void TestGPUGradients(){
			AdaptiveDropoutLayer<Dtype> layer(layer_param_);
			layer.SetUp(bottom_vec_, top_vec_);
			Caffe::set_mode(Caffe::GPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_vec_, top_vec_);
		}

	 protected:
		void SetUp(){
			vector<int> bottom_shape;
			bottom_shape.push_back(2);
			bottom_shape.push_back(1);
			bottom_shape.push_back(2);
			bottom_shape.push_back(2);
			bottom_->Reshape(bottom_shape);

			FillerParameter filler_param;
			filler_param.set_value(1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(bottom_);

			bottom_vec_.push_back(bottom_);
			top_vec_.push_back(top_);

			//set layer parameter
			layer_param_.mutable_adaptive_dropout_param()->mutable_weight_filler()->set_type("uniform");
			layer_param_.mutable_adaptive_dropout_param()->mutable_weight_filler()->set_min(-0.25);
			layer_param_.mutable_adaptive_dropout_param()->mutable_weight_filler()->set_max(0.25);
			layer_param_.mutable_adaptive_dropout_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_adaptive_dropout_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_adaptive_dropout_param()->set_num_output(3);

			shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
				this->layer_param_.adaptive_dropout_param().weight_filler()));
			
		}

		Blob<Dtype>* bottom_;
		Blob<Dtype>* top_;
		vector<Blob<Dtype>*> bottom_vec_;
		vector<Blob<Dtype>*> top_vec_;
		vector<bool> propagate_down;
		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	google::InitGoogleLogging(*argv);
	google::SetStderrLogging(0);
	caffe::AdaptiveDropoutTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUGradients();
	test.TestGPUForward();
	test.TestGPUGradients();
}
