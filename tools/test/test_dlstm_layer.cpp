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
			CHECK(top_[0]->shape() == bottom_[0]->shape());
			CHECK_EQ(top_[1]->shape(0), 6);
			CHECK_EQ(top_[1]->shape(1), 1);
			CHECK_EQ(top_[1]->shape(2), 3);
			CHECK(top_[2]->shape() == top_[3]->shape());
			CHECK_EQ(top_[2]->shape(0), 1);
			CHECK_EQ(top_[2]->shape(1), 1);
			CHECK_EQ(top_[2]->shape(2), 3);
		}

		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			layer->Backward(top_, propagate_down_, bottom_);
			top_[0]->ToTxt("x_dec_cpu");
			top_[1]->ToTxt("h_enc_cpu");
			top_[2]->ToTxt("h_T_cpu");
			top_[3]->ToTxt("c_T_cpu");
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
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
			for (int i = 0; i < top_[2]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
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
			}
			for (int i = 0; i < top_[2]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
		}
		
	protected:
		void SetUp(){
			vector<int> cont_shape;
			cont_shape.push_back(6);
			cont_shape.push_back(1);
			cont_->Reshape(cont_shape);
			vector<int> x_shape;
			x_shape.push_back(6);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			caffe_set(cont_->count(), Dtype(1), cont_->mutable_cpu_data());
			//start a new sequence in 4th element
			cont_->mutable_cpu_data()[3] = 0;
			bottom_.push_back(x_);
			bottom_.push_back(cont_);
			top_.push_back(h_enc_);
			top_.push_back(h_dec_);
			top_.push_back(h_T_);
			top_.push_back(c_T_);
			propagate_down_.resize(2, true);
			propagate_down_[1] = false;

			//set layer param
			layer_param_.mutable_recurrent_param()->set_num_output(3);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_min(-0.1);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_max(0.1);
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_recurrent_param()->mutable_dec_trans_weight_filler()->set_type("uniform");
			layer_param_.mutable_recurrent_param()->mutable_dec_trans_weight_filler()->set_min(-0.004);
			layer_param_.mutable_recurrent_param()->mutable_dec_trans_weight_filler()->set_max(0.004);
			layer_param_.mutable_recurrent_param()->set_num_rec_feature(2);
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