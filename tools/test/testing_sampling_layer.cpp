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

namespace caffe{
	template <typename Dtype>
	class SamplingLayerTest{
	public:
		SamplingLayerTest() :mu_(new Blob<Dtype>()), sigma_(new Blob<Dtype>()),
			gauss_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestCPUForward(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> sample_layer(new SamplingLayer<Dtype>(layer_param));
			sample_layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::CPU);
			CHECK(bottom_[0]->shape() == top_[0]->shape());
			sample_layer->Forward(bottom_, top_);
			const Dtype* mu_data = bottom_[0]->cpu_data();
			const Dtype* sigma_data = bottom_[1]->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			for (int i = 0; i < bottom_[0]->count(); i++){
				//expect sampling data to near the average data
				EXPECT_NEAR(mu_data[i], top_data[i], 10 * sigma_data[i]);
			}
			Dtype first_data = top_data[0];
			Dtype last_data = top_data[top_[0]->count() - 1];
			//do another forward pass, fill in the top blob with gaussian samples again
			//check that we have different values
			sample_layer->Forward(bottom_, top_);
			EXPECT_NE(first_data, top_[0]->cpu_data()[0]);
			EXPECT_NE(last_data, top_[0]->cpu_data()[top_[0]->count() - 1]);
		}
		
		void TestGPUForward(){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> sample_layer(layer_param);
			sample_layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			CHECK(bottom_[0]->shape() == top_[0]->shape());
			sample_layer->Forward(bottom_, top_);
			const Dtype* mu_data = bottom_[0]->cpu_data();
			const Dtype* sigma_data = bottom_[1]->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			for (int i = 0; i < bottom_[0]->count(); i++){
				//expect sampling data to near the average data
				EXPECT_NEAR(mu_data[i], top_data[i], 10 * sigma_data[i]);
			}
			Dtype first_data = top_data[0];
			Dtype last_data = top_data[top_[0]->count() - 1];
			//do another forward pass, fill in the top blob with gaussian samples again
			//check that we have different values
			sample_layer->Forward(bottom_, top_);
			EXPECT_NE(first_data, top_[0]->cpu_data()[0]);
			EXPECT_NE(last_data, top_[0]->cpu_data()[top_[0]->count() - 1]);
		}

		void TestCPUGradients(){
			LayerParameter layer_param;
			SamplingLayer<Dtype> sample_layer(layer_param);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&sample_layer, bottom_, top_);
		}

		void TestGPUGradients(){
			LayerParameter layer_param;
			SamplingLayer<Dtype> sample_layer(layer_param);
			Caffe::set_mode(Caffe::GPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&sample_layer, bottom_, top_);
		}

	protected:
		void SetUp(){
			mu_->Reshape(2, 1, 2, 3);
			sigma_->ReshapeLike(*mu_);
			gauss_->ReshapeLike(*mu_);
			FillerParameter filler_param;
			filler_param.set_mean(0.);
			filler_param.set_std(1.);
			GaussianFiller<Dtype> gaussian_filler(filler_param);
			gaussian_filler.Fill(mu_);
			gaussian_filler.Fill(sigma_);
			mu_->mutable_cpu_data()[2] = 3.;
			mu_->mutable_cpu_data()[5] = 4.;
			sigma_->mutable_cpu_data()[1] = 1.;
			sigma_->mutable_cpu_data()[4] = 2.;
			//make sure sigma is larger than 0.
			caffe_abs(mu_->count(), sigma_->cpu_data(), sigma_->mutable_cpu_data());
			bottom_.push_back(mu_);
			bottom_.push_back(sigma_);
			top_.push_back(gauss_);
			propagate_down.resize(2, true);
		}

		Blob<Dtype>* mu_;
		Blob<Dtype>* sigma_;
		Blob<Dtype>* gauss_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down;
	};
}

int main(int argc, char** argv){
	caffe::SamplingLayerTest<float> test;
	test.TestCPUForward();
	test.TestCPUGradients();
	test.TestGPUGradients();
}