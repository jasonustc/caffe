#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/sequence_layers.hpp"

float sigmoid_test(float input){
	return 1. / (1. + exp(-input));
}

float tanh_test(float input){
	return 2. * sigmoid_test(2. * input) - 1.;
}

namespace caffe{
	template <typename Dtype>
	class DLSTMTest{
	public:
		DLSTMTest() : x_(new Blob<Dtype>()), h_dec_(new Blob<Dtype>()), c_T_(new Blob<Dtype>()),
			h_T_(new Blob<Dtype>()){
			this->SetUp();
		}

		~DLSTMTest(){  delete x_; delete h_dec_; delete c_T_; delete h_T_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 4);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 2);
		}

		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			top_[0]->ToTxt("h_dec_cpu");
			const Dtype* x_data = x_->cpu_data();
			const Dtype* h_T_data = h_T_->cpu_data();
			const Dtype* c_T_data = c_T_->cpu_data();
			Dtype g_1_i = tanh_test(h_T_data[0] * 0.1 + 0.1 * h_T_data[1] + 0.1 * x_data[3] + 0.l * x_data[2]);
			Dtype i_1_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_T_data[0] + 0.1 * h_T_data[1]);
			Dtype f_1_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.l * h_T_data[0] + 0.1 * h_T_data[1]);
			Dtype o_1_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_T_data[0] + 0.1 * h_T_data[1]);
			Dtype c_0_i = c_T_data[0];
			Dtype c_1_i = g_1_i * i_1_i + f_1_i * c_0_i;
			Dtype h_1_i = o_1_i * tanh_test(c_1_i);
			EXPECT_NEAR(h_1_i, top_[0]->cpu_data()[2], 1e-3);


			Dtype g_2_i = tanh_test(h_1_i * 0.1 + 0.1 * x_data[1] + h_1_i * 0.1 +  0.1 * x_data[0]);
			Dtype i_2_i = sigmoid_test(0.1 * x_data[1] + 0.1 * x_data[0] + 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype f_2_i = sigmoid_test(0.1 * x_data[1] + 0.1 * x_data[0] + 0.l * h_1_i + 0.1 * h_1_i);
			Dtype o_2_i = sigmoid_test(0.1 * x_data[1] + 0.1 * x_data[0] + 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype c_2_i = g_2_i * i_2_i + f_2_i * c_1_i;
			Dtype h_2_i = o_2_i * tanh_test(c_2_i);
			EXPECT_NEAR(h_2_i, top_[0]->cpu_data()[0], 1e-3);
		}

		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			top_[0]->ToTxt("h_dec_gpu");
		}

		void TestCPUGradients(){
			DLSTMLayer<Dtype> layer(layer_param_);
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
				checker.CheckGradientSingle(&layer, bottom_, top_, 1, 0, i);
				checker.CheckGradientSingle(&layer, bottom_, top_, 2, 0, i);
			}
		}

		void TestGPUGradients(){
			DLSTMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
				checker.CheckGradientSingle(&layer, bottom_, top_, 1, 0, i);
				checker.CheckGradientSingle(&layer, bottom_, top_, 2, 0, i);
			}
		}
		
	protected:
		void SetUp(){
			vector<int> h_shape;
			h_shape.push_back(2);
			h_shape.push_back(1);
			h_shape.push_back(2);
			h_T_->Reshape(h_shape);
			c_T_->Reshape(h_shape);
			vector<int> x_shape;
			x_shape.push_back(4);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			filler.Fill(h_T_);
			filler.Fill(c_T_);
//			h_T_->mutable_cpu_data()[1] = 0.5;
//			h_T_->mutable_cpu_data()[3] = 1;
//			c_T_->mutable_cpu_data()[0] = 1.5;
//			c_T_->mutable_cpu_data()[2] = 1;
			bottom_.push_back(x_);
			bottom_.push_back(h_T_);
			bottom_.push_back(c_T_);
			top_.push_back(h_dec_);
			propagate_down_.resize(3, true);

			//set layer param
			layer_param_.mutable_recurrent_param()->set_num_output(2);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_type("constant");
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_value(0.1);
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_recurrent_param()->set_sequence_length(2);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* h_dec_;

		Blob<Dtype>* h_T_;
		Blob<Dtype>* c_T_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	caffe::DLSTMTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUGradients();
//	test.TestGPUForward();
//	test.TestGPUGradients();
	return 0;
}