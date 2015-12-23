#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	class PoolingLayerTest{
	public:
		PoolingLayerTest() : input_map_(new Blob<Dtype>()), 
			output_map_(new Blob<Dtype>()){
			this->SetUp();
		}
		~PoolingLayerTest(){ delete input_map_; delete output_map_; }
		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new PoolingLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::CPU);
			layer->SetUp(bottom_, top_);
			bottom_[0]->ToTxt("bottom_data");
			CHECK_EQ(top_[0]->num(), 2);
			CHECK_EQ(top_[0]->channels(), 1);
			CHECK_EQ(top_[0]->height(), 1);
			CHECK_EQ(top_[0]->width(), 2);
		}
		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new PoolingLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::CPU);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* top_data = top_[0]->cpu_data();
			const Dtype* bottom_data = bottom_[0]->cpu_data();
			CHECK_EQ(top_data[0], bottom_data[0]);
			CHECK_EQ(top_data[1], bottom_data[2]);
			CHECK_EQ(top_data[2], bottom_data[7]);
			CHECK_EQ(top_data[3], bottom_data[11]);
		}
		void TestCPUBackward(){
			PoolingLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			layer.SetUp(bottom_, top_);
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}
		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new PoolingLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::GPU);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* top_data = top_[0]->cpu_data();
			const Dtype* bottom_data = bottom_[0]->cpu_data();
			CHECK_EQ(top_data[0], bottom_data[0]);
			CHECK_EQ(top_data[1], bottom_data[2]);
			CHECK_EQ(top_data[2], bottom_data[7]);
			CHECK_EQ(top_data[3], bottom_data[11]);
		}
		void TestGPUBackward(){
			PoolingLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			layer.SetUp(bottom_, top_);
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}
	private:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(2);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_shape.push_back(3);
			input_map_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(input_map_);
			Dtype* x_data = input_map_->mutable_cpu_data();
			x_data[0] = (Dtype)0;
			x_data[2] = (Dtype)-1;
			x_data[7] = (Dtype)0.5;
			x_data[11] = (Dtype)0.1;
			bottom_.push_back(input_map_);
			top_.push_back(output_map_);
			propagate_down_.push_back(true);
			//set layer parameter
			layer_param_.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_MIN);
			layer_param_.mutable_pooling_param()->set_kernel_size(2);
			layer_param_.mutable_pooling_param()->set_stride(1);
		}
		LayerParameter layer_param_;
		Blob<Dtype>* input_map_;
		Blob<Dtype>* output_map_;
		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
		vector<bool> propagate_down_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	caffe::PoolingLayerTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUBackward();
	test.TestGPUForward();
	test.TestGPUBackward();
	return 0;
}