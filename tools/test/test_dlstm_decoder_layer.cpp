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
		DLSTMTest() : cont_(new Blob<Dtype>()), x_(new Blob<Dtype>()),
			h_enc_(new Blob<Dtype>()), h_dec_(new Blob<Dtype>()), c_T_(new Blob<Dtype>()),
			h_T_(new Blob<Dtype>()){
			this->SetUp();
		}

		~DLSTMTest(){ delete cont_; delete x_; delete h_enc_; delete h_dec_; delete c_T_; delete h_T_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK(top_[0]->shape() == top_[1]->shape());
			CHECK_EQ(top_[1]->shape(0), 4);
			CHECK_EQ(top_[1]->shape(1), 1);
			CHECK_EQ(top_[1]->shape(2), 2);
			CHECK(top_[2]->shape() == top_[3]->shape());
			CHECK_EQ(top_[2]->shape(0), 1);
			CHECK_EQ(top_[2]->shape(1), 1);
			CHECK_EQ(top_[2]->shape(2), 2);
		}

		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			layer->Backward(top_, propagate_down_, bottom_);
			top_[0]->ToTxt("h_dec_cpu");
			top_[1]->ToTxt("h_enc_cpu");
			top_[2]->ToTxt("h_T_cpu");
			top_[3]->ToTxt("c_T_cpu");
			const Dtype* x_data = x_->cpu_data();
			Dtype g_1_i = tanh_test(0.1 * x_data[0] + 0.1 * x_data[1]);
			Dtype i_1_i = sigmoid_test(0.1 * x_data[0] + 0.1 * x_data[1] + 0.1 * 0 + 0.1 * 0);
			Dtype f_1_i = sigmoid_test(0.1 * x_data[0] + 0.1 * x_data[1] + 0.l * 0 + 0.1 * 0);
			Dtype o_1_i = sigmoid_test(0.1 * x_data[0] + 0.1 * x_data[1] + 0.1 * 0 + 0.1 * 0);
			Dtype c_0_i = 0;
			Dtype c_1_i = g_1_i * i_1_i + f_1_i * c_0_i;
			Dtype h_1_i = o_1_i * tanh_test(c_1_i);
			EXPECT_NEAR(h_1_i, top_[1]->cpu_data()[0], 1e-3);

			Dtype g_2_i = tanh_test(0.1 * x_data[2] + 0.1 * x_data[3]+ 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype i_2_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3]+ 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype f_2_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3]+ 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype o_2_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_1_i + 0.1 * h_1_i);
			Dtype c_2_i = g_2_i * i_2_i + f_2_i * c_1_i;
			Dtype h_2_i = o_2_i * tanh_test(c_2_i);
			EXPECT_NEAR(h_2_i, top_[1]->cpu_data()[2], 1e-3);

			Dtype g_3_i = tanh_test(0.1 * x_data[4] + 0.1 * x_data[5]+ 0.1 * 0 + 0.1 * 0);
			Dtype i_3_i = sigmoid_test(0.1 * x_data[4] + 0.1 * x_data[5]+ 0.1 * 0 + 0.1 * 0);
			Dtype f_3_i = sigmoid_test(0.1 * x_data[4] + 0.1 * x_data[5]+ 0.1 * 0 + 0.1 * 0);
			Dtype o_3_i = sigmoid_test(0.1 * x_data[4] + 0.1 * x_data[5] + 0.1 * 0 + 0.1 * 0);
			Dtype c_3_i = g_3_i * i_3_i + f_3_i * 0;
			Dtype h_3_i = o_3_i * tanh_test(c_3_i);
			EXPECT_NEAR(h_3_i, top_[1]->cpu_data()[4], 1e-3);

			Dtype g_4_i = tanh_test(0.1 * x_data[6] + 0.1 * x_data[7]+ 0.1 * h_3_i + 0.1 * h_3_i);
			Dtype i_4_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7]+ 0.1 * h_3_i + 0.1 * h_3_i);
			Dtype f_4_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7]+ 0.1 * h_3_i + 0.1 * h_3_i);
			Dtype o_4_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7] + 0.1 * h_3_i + 0.1 * h_3_i);
			Dtype c_4_i = g_4_i * i_4_i + f_4_i * c_3_i;
			Dtype h_4_i = o_4_i * tanh_test(c_4_i);
			EXPECT_NEAR(h_4_i, top_[1]->cpu_data()[6], 1e-3);

			Dtype g_5_i = tanh_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_2_i + 0.1 * h_2_i);
			Dtype i_5_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_2_i + 0.1 * h_2_i);
			Dtype f_5_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_2_i + 0.1 * h_2_i);
			Dtype o_5_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_2_i + 0.1 * h_2_i);
			Dtype c_5_i = g_5_i * i_5_i + f_5_i * c_2_i;
			Dtype h_5_i = o_5_i * tanh_test(c_5_i);
			EXPECT_NEAR(h_5_i, top_[0]->cpu_data()[2], 1e-3);

			Dtype g_6_i = tanh_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_5_i + 0.1 * h_5_i);
			Dtype i_6_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_5_i + 0.1 * h_5_i);
			Dtype f_6_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_5_i + 0.1 * h_5_i);
			Dtype o_6_i = sigmoid_test(0.1 * x_data[2] + 0.1 * x_data[3] + 0.1 * h_5_i + 0.1 * h_5_i);
			Dtype c_6_i = g_6_i * i_6_i + f_6_i * c_5_i;
			Dtype h_6_i = o_6_i * tanh_test(c_6_i);
			EXPECT_NEAR(h_6_i, top_[0]->cpu_data()[0], 1e-3);

			Dtype g_7_i = tanh_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_4_i + 0.1 * h_4_i);
			Dtype i_7_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_4_i + 0.1 * h_4_i);
			Dtype f_7_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_4_i + 0.1 * h_4_i);
			Dtype o_7_i = sigmoid_test(0.1 * 0 + 0.1 * 0 + 0.1 * h_4_i + 0.1 * h_4_i);
			Dtype c_7_i = g_7_i * i_7_i + f_7_i * c_4_i;
			Dtype h_7_i = o_7_i * tanh_test(c_7_i);
			EXPECT_NEAR(h_7_i, top_[0]->cpu_data()[6], 1e-3);

			Dtype g_8_i = tanh_test(0.1 * x_data[6] + 0.1 * x_data[7] + 0.1 * h_7_i + 0.1 * h_7_i);
			Dtype i_8_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7] + 0.1 * h_7_i + 0.1 * h_7_i);
			Dtype f_8_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7] + 0.1 * h_7_i + 0.1 * h_7_i);
			Dtype o_8_i = sigmoid_test(0.1 * x_data[6] + 0.1 * x_data[7] + 0.1 * h_7_i + 0.1 * h_7_i);
			Dtype c_8_i = g_8_i * i_8_i + f_8_i * c_7_i;
			Dtype h_8_i = o_8_i * tanh_test(c_8_i);
			EXPECT_NEAR(h_8_i, top_[0]->cpu_data()[4], 1e-3);
		}

		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			layer->Backward(top_, propagate_down_, bottom_);
			top_[0]->ToTxt("x_dec_gpu");
			top_[1]->ToTxt("h_enc_gpu");
			top_[2]->ToTxt("h_T_gpu");
			top_[3]->ToTxt("c_T_gpu");
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
			for (int i = 0; i < top_[1]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 1, i);
			}
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
//  		//c_T and h_T is silenced, so the gradient will be not correct
//			for (int i = 0; i < top_[2]->count(); i++){
//				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
//			}
		}

		void TestGPUGradients(){
			DLSTMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//c_T and h_T is silenced, so the gradient will be not correct
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
			for (int i = 0; i < top_[1]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 1, i);
			}
		}
		
	protected:
		void SetUp(){
			vector<int> cont_shape;
			cont_shape.push_back(4);
			cont_shape.push_back(1);
			cont_->Reshape(cont_shape);
			vector<int> x_shape;
			x_shape.push_back(4);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_type("constant");
			filler_param.set_value(0);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			x_->mutable_cpu_data()[0] = 1;
			x_->mutable_cpu_data()[2] = 2;
			x_->mutable_cpu_data()[4] = 3;
			x_->mutable_cpu_data()[8] = 4;
			caffe_set(cont_->count(), Dtype(1), cont_->mutable_cpu_data());
			//start a new sequence in 2th element
			cont_->mutable_cpu_data()[2] = 0;
			cont_->ToTxt("cont_");
			bottom_.push_back(x_);
			bottom_.push_back(cont_);
			top_.push_back(h_enc_);
			top_.push_back(h_dec_);
			top_.push_back(h_T_);
			top_.push_back(c_T_);
			propagate_down_.resize(2, true);
			propagate_down_[1] = false;

			//set layer param
			layer_param_.mutable_recurrent_param()->set_num_output(2);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_type("constant");
//			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_min(-0.1);
//			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_max(0.1);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_value(0.1);
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_recurrent_param()->set_sequence_length(2);
			layer_param_.mutable_recurrent_param()->set_decode(true);
		}

		Blob<Dtype>* cont_;
		Blob<Dtype>* x_;

		Blob<Dtype>* h_enc_;
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
	test.TestGPUForward();
	test.TestGPUGradients();
	return 0;
}