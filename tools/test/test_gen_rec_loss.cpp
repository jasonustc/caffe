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
	class GenRecLossLayerTest{
	public:
		GenRecLossLayerTest() : mu_(new Blob<Dtype>()), sigma_(new Blob<Dtype>()),
			x_(new Blob<Dtype>()), top_blob_(new Blob<Dtype>()){
			this->SetUp();
		}
		~GenRecLossLayerTest(){ delete mu_; delete sigma_; delete x_; delete top_blob_; }
	public:
		void TestSetUp(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> layer(new GenRecLossLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
			//loss layer only has one output
			EXPECT_EQ(top_blob_->num(), 1);
			EXPECT_EQ(top_blob_->channels(), 1);
			EXPECT_EQ(top_blob_->height(), 1);
			EXPECT_EQ(top_blob_->width(), 1);
		}

		void TestCPUForward(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> layer(new GenRecLossLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(bottom_, top_);
			EXPECT_NEAR(true_loss_, top_[0]->cpu_data()[0], 1e-4);
		}

		void TestGPUForward(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> layer(new GenRecLossLayer<Dtype>(layer_param));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			EXPECT_NEAR(true_loss_, top_[0]->cpu_data()[0], 1e-4);
		}

		void TestCPUGradients(){
			LayerParameter layer_param;
			GenRecLossLayer<Dtype> layer(layer_param);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}

		void TestGPUGradients(){
			LayerParameter layer_param;
			GenRecLossLayer<Dtype> layer(layer_param);
			Caffe::set_mode(Caffe::GPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}

	protected:
		void SetUp(){
			mu_->Reshape(2, 1, 2, 3);
			sigma_->ReshapeLike(*mu_);
			x_->ReshapeLike(*mu_);
			FillerParameter filler_param;
			filler_param.set_mean(0.);
			filler_param.set_std(1.);
			GaussianFiller<Dtype> gaussian_filler(filler_param);
			gaussian_filler.Fill(mu_);
			gaussian_filler.Fill(sigma_);
			gaussian_filler.Fill(x_);
			mu_->mutable_cpu_data()[3] = Dtype(1.);
			sigma_->mutable_cpu_data()[2] = Dtype(3.);
			x_->mutable_cpu_data()[0] = Dtype(2.5);
			x_->mutable_cpu_data()[4] = Dtype(-2);
			bottom_.push_back(mu_);
			bottom_.push_back(sigma_);
			bottom_.push_back(x_);
			top_.push_back(top_blob_);
			//set propagate_down
			propagate_down_.resize(true, 3);
			//do not propagate gradient to x data
			propagate_down_[2] = false;
			//get true loss by raw cpu code
			GetTrueLoss();
		}

		void GetTrueLoss(){
			const int count = x_->count();
			const Dtype* x_data = x_->cpu_data();
			const Dtype* mu_data = mu_->cpu_data();
			const Dtype* sigma_data = sigma_->cpu_data();
			true_loss_ = 0.;
			for (int i = 0; i < count; i++){
				true_loss_ += (x_data[i] - mu_data[i]) * (x_data[i] - mu_data[i]) /
					(sigma_data[i] * sigma_data[i]) + log(sigma_data[i]);
			}
			true_loss_ += Dtype(0.5) * Dtype(bottom_[0]->count(2)) * log(2 * Dtype(PI));
			true_loss_ /= x_->num();
		}

		Blob<Dtype>* mu_;
		Blob<Dtype>* sigma_;
		Blob<Dtype>* x_;
		Blob<Dtype>* top_blob_;
		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
		vector<bool> propagate_down_;

		Dtype true_loss_;
	};
}

int main(int argc, char** argv){
	caffe::GenRecLossLayerTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestGPUForward();
	test.TestCPUGradients();
	test.TestGPUGradients();
}
