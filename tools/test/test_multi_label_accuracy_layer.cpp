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
	class MultiLabelAccuracyTest{
	public:
		MultiLabelAccuracyTest() : pred_(new Blob<Dtype>()), label_(new Blob<Dtype>()),
		accuracy_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new MultiLabelAccuracyLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->count(), 5);
		}

		void TestForward(Caffe::Brew caffe_mode){
			shared_ptr<Layer<Dtype>> layer(new MultiLabelAccuracyLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			Caffe::set_mode(caffe_mode);
			layer->SetUp(bottom_, top_);
			//process the full sequence in a single batch
			layer->Forward(bottom_, top_);

			// Copy the inputs and outputs to reuse/check them later
			EXPECT_NEAR(0.5, top_[0]->cpu_data()[0], 1e-4);
			EXPECT_NEAR(0.5, top_[0]->cpu_data()[1], 1e-4);
			EXPECT_NEAR(0.5, top_[0]->cpu_data()[2], 1e-4);
			EXPECT_NEAR(0.5, top_[0]->cpu_data()[3], 1e-4);
			EXPECT_NEAR(0.5, top_[0]->cpu_data()[4], 1e-4);
		}

	protected:
		void SetUp(){
			vector<int> pred_shape(2, 4);
			pred_shape[0] = 2;
			pred_->Reshape(pred_shape);
			label_->Reshape(pred_shape);
			FillerParameter filler_param;
			filler_param.set_type("constant");
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(pred_);
			pred_->mutable_cpu_data()[0] = -1;
			pred_->mutable_cpu_data()[5] = -0.5;
			pred_->mutable_cpu_data()[2] = -0.4;
			pred_->mutable_cpu_data()[6] = -0.4;
			label_->mutable_cpu_data()[0] = 1;
			label_->mutable_cpu_data()[3] = 1;
			label_->mutable_cpu_data()[5] = -1;
			label_->mutable_cpu_data()[7] = -1;
			bottom_.push_back(pred_);
			bottom_.push_back(label_);
			top_.push_back(accuracy_);
		}

		LayerParameter layer_param_;

		Blob<Dtype>* pred_;
		Blob<Dtype>* label_;

		Blob<Dtype>* accuracy_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
	};
}

int main(int argc, char** argv){
	caffe::MultiLabelAccuracyTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	return 0;
}