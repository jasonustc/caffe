#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/added_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe{

	extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

	template <typename Dtype>
	class LocalLayerTest :public ::testing::Test{
	protected:
		LocalLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){}
		virtual void SetUp(){
			blob_bottom_->Reshape(2, 3, 6, 4);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~LocalLayerTest(){ delete blob_bottom_; delete blob_top_; }
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	typedef ::testing::Types<float, double> Dtypes;
	TYPED_TEST_CASE(LocalLayerTest, Dtypes);

	TYPED_TEST(LocalLayerTest, TestSetup){
		LayerParameter layer_param;
		LocalParameter* convolution_param =
			layer_param.mutable_local_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		convolution_param->set_num_output(4);
		shared_ptr<Layer<TypeParam>> layer(
			new LocalLayer<TypeParam>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), 2);
		EXPECT_EQ(this->blob_top_->channels(), 4);
		EXPECT_EQ(this->blob_top_->height(), 2);
		EXPECT_EQ(this->blob_top_->width(), 1);
		convolution_param->set_num_output(3);
		layer.reset(new LocalLayer<TypeParam>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), 2);
		EXPECT_EQ(this->blob_top_->channels(), 3);
		EXPECT_EQ(this->blob_top_->height(), 2);
		EXPECT_EQ(this->blob_top_->width(), 1);
	}

	TYPED_TEST(LocalLayerTest, TestCPUSimpleConvolution){
		//We will simply see if the convolution layer carries out averaging well.
		FillerParameter filler_param;
		filler_param.set_value(1.);
		ConstantFiller<TypeParam> filler(filler_param);
		filler.Fill(this->blob_bottom_);
		LayerParameter layer_param;
		LocalParameter* convolution_param =
			layer_param.mutable_local_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(1);
		convolution_param->set_num_output(1);
		convolution_param->mutable_weight_filler()->set_type("constant");
		convolution_param->mutable_weight_filler()->set_value(1);
		convolution_param->mutable_bias_filler()->set_type("constant");
		convolution_param->mutable_bias_filler()->set_value(0.1);

		shared_ptr<Layer<TypeParam>> layer(
			new LocalLayer<TypeParam>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		Caffe::set_mode(Caffe::CPU);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		//After the convolution, the output should all have output values 27.1
		const TypeParam* top_data = this->blob_top_->cpu_data();
		for (int n = 0; n < this->blob_top_->num(); n++){
			for (int c = 0; c < this->blob_top_->channels(); c++){
				for (int h = 0; h < this->blob_top_->height(); h++){
					for (int w = 0; w < this->blob_top_->width(); w++){
						int idx = h*this->blob_top_->width() + w;
						EXPECT_NEAR(*(top_data + this->blob_top_->offset(n, c, h, w)), idx * 27 + 0.1, 1e-4);
					}
				}
			}
		}
	}

	TYPED_TEST(LocalLayerTest, TestGPUSimpleConvolution){
		//we will simply see if the convolution layer carries out averaging well.
		FillerParameter filler_param;
		filler_param.set_value(1.);
		ConstantFiller<TypeParam> filler(filler_param);
		filler.Fill(this->blob_bottom_);
		LayerParameter layer_param;
		LocalParameter* convolution_param =
			layer_param.mutable_local_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		convolution_param->set_num_output(4);
		convolution_param->mutable_weight_filler()->set_type("constant");
		convolution_param->mutable_weight_filler()->set_value(1);
		convolution_param->mutable_bias_filler()->set_type("constant");
		convolution_param->mutable_bias_filler()->set_value(0.1);
		shared_ptr<Layer<TypeParam>> layer(
			new LocalLayer<TypeParam>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		Caffe::set_mode(Caffe::GPU);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		//After the convolution, the output should all have output values 27.1
		const TypeParam* top_data = this->blob_top_->cpu_data();
		for (int n = 0; n < this->blob_top_->num(); n++){
			for (int c = 0; c < this->blob_top_->channels(); c++){
				for (int h = 0; h < this->blob_top_->height(); h++){
					for (int w = 0; w < this->blob_top_->width(); w++){
						int idx = h*this->blob_top_->width() + w;
						EXPECT_NEAR(*(top_data + this->blob_top_->offset(n, c, h, w)), idx * 27 + 0.1, 1e-4);
					}
				}
			}
		}
	}

	TYPED_TEST(LocalLayerTest, TestCPUGradient){
		LayerParameter layer_param;
		LocalParameter* convolution_param =
			layer_param.mutable_local_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		convolution_param->set_num_output(2);
		convolution_param->mutable_weight_filler()->set_type("gaussian");
		convolution_param->mutable_bias_filler()->set_type("gaussian");
		Caffe::set_mode(Caffe::CPU);
		LocalLayer<TypeParam> layer(layer_param);
		GradientChecker<TypeParam> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
	}

	TYPED_TEST(LocalLayerTest, TestGPUGradient){
		LayerParameter layer_param;
		LocalParameter* convolution_param =
			layer_param.mutable_local_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		convolution_param->set_num_output(2);
		convolution_param->mutable_weight_filler()->set_type("gaussian");
		convolution_param->mutable_bias_filler()->set_type("gaussian");
		Caffe::set_mode(Caffe::GPU);
		LocalLayer<TypeParam> layer(layer_param);
		GradientChecker<TypeParam> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
	}
}

