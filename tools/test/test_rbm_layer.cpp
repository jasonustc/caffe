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
			Dtype* top_diff = top_[0]->mutable_cpu_diff();
			top_diff[2] = 1;
			top_diff[4] = 4;
			caffe_set(top_[0]->count(), Dtype(0.1), top_diff);
			layer.Backward(top_, propagate_down_, bottom_);
			const Dtype* pos_v_data = layer.pos_v_.cpu_data();
			const Dtype* neg_v_data = layer.neg_v_.cpu_data();
			const Dtype* pos_h_data = layer.pos_h_.cpu_data();
			const Dtype* neg_h_data = layer.neg_h_.cpu_data();
			const int count_v = layer.pos_v_.count(1);
			const int count_h = layer.pos_h_.count(1);
			const int num = layer.pos_v_.num();
			const Dtype* weight_diff = layer.blobs()[0]->cpu_diff();
			const Dtype* h_bias_diff = layer.blobs()[1]->cpu_diff();
			const Dtype* v_bias_diff = layer.blobs()[2]->cpu_diff();
			for (int i = 0; i < count_h; i++){
				for (int j = 0; j < count_v; j++){
					Dtype pos_en = 0;
					Dtype neg_en = 0;
					for (int n = 0; n < num; n++){
						pos_en += pos_v_data[n * count_v + j] * pos_h_data[n * count_h + i];
						neg_en += neg_v_data[n * count_v + j] * neg_h_data[n * count_h + i];
					}
					Dtype diff = (pos_en - neg_en) / num;
					LOG(INFO) << weight_diff[i * count_v + j];
					EXPECT_NEAR(weight_diff[i * count_v + j], diff, 1e-3);
				}
			}
			for (int i = 0; i < count_h; i++){
				Dtype diff = 0;
				for (int n = 0; n < num; n++){
					diff += pos_h_data[n * count_h + i] - neg_h_data[n * count_h + i];
				}
				diff /= num;
				EXPECT_NEAR(h_bias_diff[i], diff, 1e-3);
			}
			for (int j = 0; j < count_v; j++){
				Dtype diff = 0;
				for (int n = 0; n < num; n++){
					diff += pos_v_data[n * count_v + j] - neg_v_data[n * count_v + j];
				}
				diff /= num;
				EXPECT_NEAR(v_bias_diff[j], diff, 1e-3);
			}
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
//			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
//			checker.CheckGradientExhaustive(&layer, bottom_, top_);
//			LOG(INFO) << top_[0]->count();
//			for (int i = 0; i < top_[0]->count(); i++){
//				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
//			}
		}

		void TestGPUGradients(){
			RBMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			layer.Forward(bottom_, top_);
			Dtype* top_diff = top_[0]->mutable_cpu_diff();
			top_diff[2] = 1;
			top_diff[4] = 4;
			caffe_set(top_[0]->count(), Dtype(0.1), top_diff);
			layer.Backward(top_, propagate_down_, bottom_);
			const Dtype* pos_v_data = layer.pos_v_.cpu_data();
			const Dtype* neg_v_data = layer.neg_v_.cpu_data();
			const Dtype* pos_h_data = layer.pos_h_.cpu_data();
			const Dtype* neg_h_data = layer.neg_h_.cpu_data();
			const int count_v = layer.pos_v_.count(1);
			const int count_h = layer.pos_h_.count(1);
			const int num = layer.pos_v_.num();
			const Dtype* weight_diff = layer.blobs()[0]->cpu_diff();
			const Dtype* h_bias_diff = layer.blobs()[1]->cpu_diff();
			const Dtype* v_bias_diff = layer.blobs()[2]->cpu_diff();
			for (int i = 0; i < count_h; i++){
				for (int j = 0; j < count_v; j++){
					Dtype pos_en = 0;
					Dtype neg_en = 0;
					for (int n = 0; n < num; n++){
						pos_en += pos_v_data[n * count_v + j] * pos_h_data[n * count_h + i];
						neg_en += neg_v_data[n * count_v + j] * neg_h_data[n * count_h + i];
					}
					Dtype diff = (pos_en - neg_en) / num;
					LOG(INFO) << weight_diff[i * count_v + j];
					EXPECT_NEAR(weight_diff[i * count_v + j], diff, 1e-3);
				}
			}
			for (int i = 0; i < count_h; i++){
				Dtype diff = 0;
				for (int n = 0; n < num; n++){
					diff += pos_h_data[n * count_h + i] - neg_h_data[n * count_h + i];
				}
				diff /= num;
				EXPECT_NEAR(h_bias_diff[i], diff, 1e-3);
			}
			for (int j = 0; j < count_v; j++){
				Dtype diff = 0;
				for (int n = 0; n < num; n++){
					diff += pos_v_data[n * count_v + j] - neg_v_data[n * count_v + j];
				}
				diff /= num;
				EXPECT_NEAR(v_bias_diff[j], diff, 1e-3);
			}
//			GradientChecker<Dtype> checker(1e-2, 1e-3);
//			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
//			checker.CheckGradientExhaustive(&layer, bottom_, top_);
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
	test.TestCPUGradients();
//	test.TestGPUForward();
//	test.TestGPUGradients();
	return 0;
}