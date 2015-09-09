#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe{

	template <typename Dtype>
	class NormLossLayerTest{
	public:
		NormLossLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}

		~NormLossLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(2, 1, 2, 2);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
			//propagate to bottom
			propagate_down.resize(this->blob_bottom_vec_.size(), true);
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		vector<bool> propagate_down;

	public:
		void TestSetup(){
			LayerParameter layer_param;
			NormLossParameter* norm_loss_param = layer_param.mutable_norm_loss_param();
			norm_loss_param->set_norm_type(NormLossParameter_NormType_L1);
			shared_ptr<Layer<Dtype>> layer(new NormLossLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->num(), 1);
			EXPECT_EQ(this->blob_top_->channels(), 1);
			EXPECT_EQ(this->blob_top_->height(), 1);
			EXPECT_EQ(this->blob_top_->width(), 1);
		}

		void TestCPUNormLoss(){
			FillerParameter filler_param;
			filler_param.set_value(1.);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			this->blob_bottom_->mutable_cpu_data()[0] = -1;
			this->blob_bottom_->mutable_cpu_data()[1] = 2;
			LayerParameter layer_param;
			NormLossParameter* norm_param =
				layer_param.mutable_norm_loss_param();
			norm_param->set_norm_type(NormLossParameter_NormType_L1);

			shared_ptr<Layer<Dtype>> layer(new NormLossLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			Caffe::set_mode(Caffe::CPU);
			layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			//After all the norm values should be equal to sum of abs values
			const Dtype* top_data = this->blob_top_->cpu_data();
			Dtype sum_norm(0);
			for (int i = 0; i < this->blob_bottom_vec_[0]->count(); i++){
				sum_norm += abs(this->blob_bottom_vec_[0]->cpu_data()[i]);
			}
			EXPECT_NEAR(sum_norm, top_data[0], 1e-4);
		}

		void TestGPUNormLoss(){
			//we will simply see if the convolution layer carries out averaging well.
			FillerParameter filler_param;
			filler_param.set_value(1.);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			this->blob_bottom_->mutable_cpu_data()[0] = 2;
			this->blob_bottom_->mutable_cpu_data()[1] = -1;
			LayerParameter layer_param;
			NormLossParameter* norm_param =
				layer_param.mutable_norm_loss_param();
			norm_param->set_norm_type(NormLossParameter_NormType_L1);
			shared_ptr<Layer<Dtype>> layer(new NormLossLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			//After the convotion, the output should all have output values 27.1
			const Dtype* top_data = this->blob_top_->cpu_data();
			Dtype sum_norm(0);
			for (int i = 0; i < this->blob_bottom_->count(); i++){
				sum_norm += abs(this->blob_bottom_vec_[0]->cpu_data()[i]);
			}
			EXPECT_NEAR(sum_norm, top_data[0], 1e-4);
		}

		void TestCPUGradient(){
			LayerParameter layer_param;
			NormLossParameter* norm_param =
				layer_param.mutable_norm_loss_param();
			norm_param->set_norm_type(NormLossParameter_NormType_L1);
	  		Caffe::set_mode(Caffe::CPU);
			NormLossLayer<Dtype> checker_layer(layer_param);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&checker_layer, this->blob_bottom_vec_, this->blob_top_vec_);
		}

		void TestGPUGradient(){
			LayerParameter layer_param;
			NormLossParameter* norm_param =
				layer_param.mutable_norm_loss_param();
			norm_param->set_norm_type(NormLossParameter_NormType_L1);
			Caffe::set_mode(Caffe::GPU);
			NormLossLayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
		}
	};
}

int main(int argc, char** argv){
	caffe::NormLossLayerTest<float> test;
	test.TestSetup();
	test.TestCPUGradient();
	test.TestGPUGradient();
	test.TestGPUNormLoss();
	test.TestCPUNormLoss();
	return 0;
}
