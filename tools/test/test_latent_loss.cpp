#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/loss_layers.hpp"

namespace caffe{
	template <typename Dtype>
	class LatentLossLayerTest{
	public:
		LatentLossLayerTest() : blob_bottom_mu_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()), blob_bottom_sigma_(new Blob<Dtype>()){
			this->SetUp();
		}
		
		~LatentLossLayerTest(){ delete blob_bottom_; delete blob_top_; }

	public:
		void TestSetUp(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> layer(new LatentLossLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			//loss layer only has one output
			EXPECT_EQ(this->blob_top_->num(), 1);
			EXPECT_EQ(this->blob_top_->channels(), 1);
			EXPECT_EQ(this->blob_top_->height(), 1);
			EXPECT_EQ(this->blob_top_->width(), 1);
		}

		void TestCPULatentLoss(){
			FillerParameter filler_param;
			filler_param.set_value(Dtype(1.));
			ConstantFiller<Dtype> const_filler(filler_param);
			const_filler.Fill(blob_bottom_mu_);
			const_filler.Fill(blob_bottom_sigma_);
			blob_bottom_vec_[0]->mutable_cpu_data()[10] = 1.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[3] = 0.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[7] = 2;
			blob_bottom_vec_[1]->mutable_cpu_data()[1] = 1.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[2] = 0.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[5] = 2;

			LayerParameter layer_param;
			shared_ptr<Blob<Dtype>> latent_layer(new LatentLossLayer<Dtype>(layer_param));
			latent_layer->SetUp(blob_bottom_vec_, blob_top_vec_);
			Caffe::set_mode(Caffe::CPU);
			latent_layer->Forward(blob_bottom_vec_, blob_top_vec_);
			Dtype true_loss;
			const int count = blob_bottom_vec_[0]->count();
			const Dtype* mu_data = blob_bottom_vec_[0]->cpu_data();
			const Dtype* sigma_data = blob_bottom_vec_[1]->cpu_data();
			for (int i = 0; i < count; i++){
				true_loss += mu_data[i] * mu_data[i] + sigma_data[i] * sigma_data[i] + log(sigma_data[i] * sigma_data[i]);
			}
			EXPECT_NEAR(true_loss, blob_top_vec_[0]->cpu_data()[0], 1e-4);
		}

		void TestGPULatentLoss(){
			FillerParameter filler_param;
			filler_param.set_value(Dtype(1.));
			const_filler.Fill(blob_bottom_mu_);
			const_filler.Fill(blob_bottom_sigma_);
			blob_bottom_vec_[0]->mutable_cpu_data()[10] = 1.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[3] = 0.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[7] = 2;
			blob_bottom_vec_[1]->mutable_cpu_data()[1] = 1.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[2] = 0.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[5] = 2;

			LayerParameter layer_param;
			shared_ptr<Blob<Dtype>> latent_layer(new LatentLossLayer<Dtype>(layer_param));
			latent_layer->SetUp(blob_bottom_vec_, blob_top_vec_);
			Caffe::set_mode(Caffe::GPU);
			latent_layer->Forward(blob_bottom_vec_, blob_top_vec_);
			Dtype true_loss;
			const int count = blob_bottom_vec_[0]->count();
			const Dtype* mu_data = blob_bottom_vec_[0]->cpu_data();
			const Dtype* sigma_data = blob_bottom_vec_[1]->cpu_data();
			for (int i = 0; i < count; i++){
				true_loss += mu_data[i] * mu_data[i] + sigma_data[i] * sigma_data[i] + log(sigma_data[i] * sigma_data[i]);
			}
			EXPECT_NEAR(true_loss, blob_top_vec_[0]->cpu_data()[0], 1e-4);
		}

		void TestCPULatentLossGradient(){
			FillerParameter filler_param;
			filler_param.set_value(Dtype(1.));
			const_filler.Fill(blob_bottom_mu_);
			const_filler.Fill(blob_bottom_sigma_);
			blob_bottom_vec_[0]->mutable_cpu_data()[10] = 1.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[3] = 0.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[7] = 2;
			blob_bottom_vec_[1]->mutable_cpu_data()[1] = 1.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[2] = 0.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[5] = 2;

			LayerParameter layer_param;
			shared_ptr<Blob<Dtype>> latent_layer(new LatentLossLayer<Dtype>(layer_param));
			latent_layer->SetUp(blob_bottom_vec_, blob_top_vec_);
			Caffe::set_mode(Caffe::CPU);
			//check(step_size, step_range,...)
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(latent_layer, blob_bottom_vec_, blob_top_vec_);
		}

		void TestGPULatentLossGradient(){
			FillerParameter filler_param;
			filler_param.set_value(Dtype(1.));
			const_filler.Fill(blob_bottom_mu_);
			const_filler.Fill(blob_bottom_sigma_);
			blob_bottom_vec_[0]->mutable_cpu_data()[10] = 1.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[3] = 0.5;
			blob_bottom_vec_[0]->mutable_cpu_data()[7] = 2;
			blob_bottom_vec_[1]->mutable_cpu_data()[1] = 1.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[2] = 0.5;
			blob_bottom_vec_[1]->mutable_cpu_data()[5] = 2;

			LayerParameter layer_param;
			shared_ptr<Blob<Dtype>> latent_layer(new LatentLossLayer<Dtype>(layer_param));
			latent_layer->SetUp(blob_bottom_vec_, blob_top_vec_);
			Caffe::set_mode(Caffe::GPU);
			//check(step_size, step_range,...)
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(latent_layer, blob_bottom_vec_, blob_top_vec_);
		}
	protected:
		void SetUp(){
			blob_bottom_mu_->Reshape(2, 1, 2, 3);
			blob_bottom_sigma_->Reshape(2, 1, 2, 3);
			//fill the value
			FillerParameter filler_param;
			//the value in constant filler
			filler_param.set_value(Dtype(1.));
			ConstantFiller<Dtype> const_filler(filler_param);
			const_filler.Fill(blob_bottom_mu_);
			const_filler.Fill(blob_bottom_sigma_);
			blob_bottom_vec_.push_back(blob_bottom_mu_);
			blob_bottom_vec_.push_back(blob_bottom_sigma_);
			blob_top_vec_.push_back(blob_top_);
			//set propagate down
			propagate_down_.resize(blob_bottom_vec_.size(), true);
		}

	protected:
		Blob<Dtype>* const blob_bottom_mu_;
		Blob<Dtype>* const blob_bottom_sigma_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		vector<bool> propagate_down_;
	};
}

int main(int argc, char** argv){
	caffe::LatentLossLayerTest<float> test;
	test.TestSetUp();
	test.TestCPULatentLoss();
	test.TestCPULatentLossGradient();
	test.TestGPULatentLoss();
	test.TestGPULatentLossGradient();
	return 0;
}
