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
	class NeighborContrastLayerTest{
	public:
		NeighborContrastLayerTest() : x_(new Blob<Dtype>()), x_contrast_(new Blob<Dtype>()){
			this->SetUp();
		}

		~NeighborContrastLayerTest(){  delete x_; delete x_contrast_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new NeighborContrastLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 2);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 2);
			CHECK_EQ(top_[0]->shape(3), 1);
		}

		//TODO: check this test code
		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new NeighborContrastLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::CPU);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			const int offset = 2 * (2 - 1);
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					Dtype buf = x_data[(i + 1) * 2 + j] - x_data[i * 2 + j];
					EXPECT_NEAR(top_data[i * 2 + j], buf, 1e-3);
				}
				x_data += offset + 2;
				top_data += offset;
			}
			top_[0]->ToTxt("top_data");
		}

		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new NeighborContrastLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::GPU);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			const int offset = 2 * (2 - 1);
			for (int i = 0; i < 2; i++){
				for (int j = 0; j < 2; j++){
					Dtype buf = x_data[(i + 1) * 2 + j] - x_data[i * 2 + j];
					EXPECT_NEAR(top_data[i * 2 + j], buf, 1e-3);
				}
				x_data += offset + 2;
				top_data += offset;
			}
		}

		void TestCPUGradients(){
			NeighborContrastLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}

		void TestGPUGradients(){
			NeighborContrastLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}
		
	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(4);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_shape.push_back(1);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			x_->mutable_cpu_data()[0] = 1;
			x_->mutable_cpu_data()[1] = 2;
			x_->mutable_cpu_data()[2] = 0.5;
			x_->mutable_cpu_data()[5] = 1.5;
			bottom_.push_back(x_);
			top_.push_back(x_contrast_);
			propagate_down_.resize(1, true);

			//set layer param
			layer_param_.mutable_recurrent_param()->set_sequence_length(2);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* x_contrast_;


		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	caffe::NeighborContrastLayerTest<float> test;
	test.TestSetUp();
	test.TestCPUForward();
	test.TestCPUGradients();
	test.TestGPUForward();
	test.TestGPUGradients();
	return 0;
}