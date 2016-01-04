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

namespace caffe{
	template <typename Dtype>
	class LSTMTest{
	public:
		LSTMTest() : cont_(new Blob<Dtype>()), x_(new Blob<Dtype>()),
			h_enc_(new Blob<Dtype>()), c_T_(new Blob<Dtype>()),
			h_T_(new Blob<Dtype>()){
			this->SetUp();
		}

		~LSTMTest(){ delete cont_; delete x_; delete h_enc_; delete c_T_; delete h_T_; }

		void TestSetUp(){
			LayerParameter layer_param;
			layer_param.mutable_recurrent_param()->set_num_output(2);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
			//layer_param.mutable_recurrent_param()->set_decode(true);
			//layer_param.mutable_recurrent_param()->set_sequence_length(2);
			shared_ptr<Layer<Dtype>> layer(new LSTMLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
//			CHECK(top_[0]->shape() == top_[1]->shape());
			CHECK_EQ(top_[0]->shape(0), 4);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 2);
//			CHECK(top_[2]->shape() == top_[3]->shape());
//			CHECK_EQ(top_[2]->shape(0), 1);
//			CHECK_EQ(top_[2]->shape(1), 1);
//			CHECK_EQ(top_[2]->shape(2), 3);
		}

		void TestCPUForward(){
			LayerParameter layer_param;
			layer_param.mutable_recurrent_param()->set_num_output(2);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
//			layer_param.mutable_recurrent_param()->set_decode(true);
			layer_param.mutable_recurrent_param()->set_sequence_length(2);
			shared_ptr<Layer<Dtype>> layer(new LSTMLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			top_[0]->ToTxt("h_dec_cpu");
//			top_[1]->ToTxt("h_enc_cpu");
//			top_[2]->ToTxt("h_T_cpu");
//			top_[3]->ToTxt("c_T_cpu");
		}

		void TestGPUForward(){
			LayerParameter layer_param;
			layer_param.mutable_recurrent_param()->set_num_output(2);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
//			layer_param.mutable_recurrent_param()->set_decode(true);
			layer_param.mutable_recurrent_param()->set_sequence_length(2);
			shared_ptr<Layer<Dtype>> layer(new LSTMLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			top_[0]->ToTxt("h_dec_gpu");
//			top_[1]->ToTxt("h_enc_gpu");
//			top_[2]->ToTxt("h_T_gpu");
//			top_[3]->ToTxt("c_T_gpu");
		}

		void TestCPUGradients(){
			LayerParameter layer_param;
			layer_param.mutable_recurrent_param()->set_num_output(2);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
//			layer_param.mutable_recurrent_param()->set_decode(true);
//			layer_param.mutable_recurrent_param()->set_sequence_length(2);
			LSTMLayer<Dtype> layer(layer_param);
			Caffe::set_mode(Caffe::CPU);
			layer.SetUp(bottom_, top_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//check_bottom, top_id, top_data_id
			checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, 0);
			checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, 1);
			checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, 2);
			checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, 3);
		}

		void TestGPUGradients(){
			LayerParameter layer_param;
			layer_param.mutable_recurrent_param()->set_num_output(2);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
//			layer_param.mutable_recurrent_param()->set_decode(true);
//			layer_param.mutable_recurrent_param()->set_sequence_length(2);
			LSTMLayer<Dtype> layer(layer_param);
			Caffe::set_mode(Caffe::GPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
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
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			caffe_set(cont_->count(), Dtype(1), cont_->mutable_cpu_data());
			//start a new sequence in 3th element
			cont_->mutable_cpu_data()[2] = 0;
			bottom_.push_back(x_);
			bottom_.push_back(cont_);
			top_.push_back(h_enc_);
//			top_.push_back(h_dec_);
			top_.push_back(h_T_);
			top_.push_back(c_T_);
			propagate_down.resize(2, true);
			propagate_down[1] = false;
		}

		Blob<Dtype>* cont_;
		Blob<Dtype>* x_;

		Blob<Dtype>* h_enc_;

		Blob<Dtype>* h_T_;
		Blob<Dtype>* c_T_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down;
	};
}

int main(int argc, char** argv){
	caffe::LSTMTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUGradients();
	test.TestGPUForward();
	test.TestGPUGradients();
	return 0;
}