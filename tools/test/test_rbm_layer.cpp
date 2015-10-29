#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/common_layers.hpp"


namespace caffe{
	float sigmoid_test(float input){
		return 1. / (1. + exp(-input));
	}
	template <typename Dtype>
	class RBMTest{
	public:
		RBMTest() : x_(new Blob<Dtype>()), h_(new Blob<Dtype>()), loss_(new Blob<Dtype>()){
			this->SetUp();
		}

		~RBMTest(){  delete x_; delete h_; delete loss_;}

		void TestSetUp(){
			shared_ptr<RBMLayer<Dtype>> layer(new RBMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 2);
			CHECK_EQ(top_[0]->shape(1), 3);
		}

		void TestCPUForward(){
			shared_ptr<RBMLayer<Dtype>> layer(new RBMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* loss_data = loss_->cpu_data();
			for (int m = 0; m < 2; m++){
				for (int k = 0; k < 2; k += 2){
					Dtype est_x = sigmoid_test(0.1 * x_data[m * 2 + k] + 0.1 * x_data[m * 2 + k + 1]);
					EXPECT_NEAR(est_x, top_[0]->cpu_data()[m * 3 + 0], 1e-3);
				}
			}

			const Dtype* pos_v = layer->pos_v_.cpu_data();
			const Dtype* neg_v = layer->neg_v_.cpu_data();
			int count_v = layer->neg_v_.count();

			Dtype loss = 0;
			for (int i = 0; i < count_v; i++){
				loss += (pos_v[i] - neg_v[i]) * (pos_v[i] - neg_v[i]);
			}

			EXPECT_NEAR(loss / Dtype(bottom_[0]->num()), top_[1]->cpu_data()[0], 1e-3);
			
			//internal variables
			layer->positive_state_h_.ToTxt("state_h_data");
			layer->pos_v_.ToTxt("pos_v_data");
			layer->pos_h_.ToTxt("pos_h_data");
		}

		void TestGPUForward(){
			shared_ptr<RBMLayer<Dtype>> layer(new RBMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* loss_data = loss_->cpu_data();
			for (int m = 0; m < 2; m++){
				for (int k = 0; k < 2; k += 2){
					Dtype est_x = sigmoid_test(0.1 * x_data[m * 2 + k] + 0.1 * x_data[m * 2 + k + 1]);
					EXPECT_NEAR(est_x, top_[0]->cpu_data()[m * 3 + 0], 1e-3);
				}
			}

			const Dtype* pos_v = layer->pos_v_.cpu_data();
			const Dtype* neg_v = layer->neg_v_.cpu_data();
			int count_v = layer->neg_v_.count();

			Dtype loss = 0;
			for (int i = 0; i < count_v; i++){
				loss += (pos_v[i] - neg_v[i]) * (pos_v[i] - neg_v[i]);
			}

			EXPECT_NEAR(loss / Dtype(bottom_[0]->num()), top_[1]->cpu_data()[0], 1e-3);
			
			//internal variables
			layer->positive_state_h_.ToTxt("state_h_data");
			layer->pos_v_.ToTxt("pos_v_data");
			layer->pos_h_.ToTxt("pos_h_data");
		}

		void TestCPUGradients(){
			RBMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::CPU);
			layer.SetUp(bottom_, top_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			layer.Forward(bottom_, top_);
			layer.Backward(top_, propagate_down_, bottom_);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
//			LOG(INFO) << top_[0]->count();
//			for (int i = 0; i < top_[0]->count(); i++){
//				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
//			}
		}

		void TestGPUGradients(){
			RBMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
//			for (int i = 0; i < top_[0]->count(); i++){
//				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
//			}
		}
		
	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(2);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			x_->mutable_cpu_data()[0] = 1;
			x_->mutable_cpu_data()[3] = 4;
			bottom_.push_back(x_);
			top_.push_back(h_);
			top_.push_back(loss_);
			propagate_down_.resize(1, true);

			//set layer param
			layer_param_.mutable_rbm_param()->set_num_output(3);
			layer_param_.mutable_rbm_param()->mutable_weight_filler()->set_type("constant");
			layer_param_.mutable_rbm_param()->mutable_weight_filler()->set_value(0.1);
			layer_param_.mutable_rbm_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_rbm_param()->mutable_bias_filler()->set_value(0.);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* h_;
		Blob<Dtype>* loss_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	caffe::RBMTest<float> test;
	test.TestSetUp();
//	test.TestCPUForward();
//	test.TestCPUGradients();
	test.TestGPUForward();
	test.TestGPUGradients();
	return 0;
}