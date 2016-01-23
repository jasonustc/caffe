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
	class MultiLabelLossTest{
	public:
		MultiLabelLossTest() : pred_(new Blob<Dtype>()), label_(new Blob<Dtype>()),
			prob_(new Blob<Dtype>()), loss_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new MultiLabelLossLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->count(), 1) << "bad";
//			CHECK(top_[1]->shape() == bottom_[0]->shape());
		}

		void TestForward(Caffe::Brew caffe_mode){
			shared_ptr<Layer<Dtype>> layer(new MultiLabelLossLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			Caffe::set_mode(caffe_mode);
			layer->SetUp(bottom_, top_);
			//process the full sequence in a single batch
			layer->Forward(bottom_, top_);

			// Copy the inputs and outputs to reuse/check them later
			const int count = bottom_[0]->count();
			Dtype* pred_data = bottom_[0]->mutable_cpu_data();
			const Dtype* label = bottom_[1]->cpu_data();
			Dtype loss = 0;
			for (int i = 0; i < count; i++){
				if (label[i] > 0 && pred_data[i] >= 0){
					Dtype prob = Dtype(1) / ((Dtype)1. + exp(-pred_data[i]));
					loss -= log(prob);
				}
				else if (label[i] > 0 && pred_data[i] < 0){
					Dtype this_loss = pred_data[i] - log(1 + exp(pred_data[i]));
					loss -= this_loss;
				}
			}
			loss /= bottom_[0]->num();
			EXPECT_NEAR(loss, top_[0]->cpu_data()[0], 1e-4);
//			for (int i = 0; i < count; i++){
//				EXPECT_NEAR(pred_data[i], top_[1]->cpu_data()[i], 1e-4);
//			}
		}

		void TestBackward(Caffe::Brew caffe_mode){
			MultiLabelLossLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(caffe_mode);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//only check gradient to bottom[0], can not backpropagate to bottom[1]
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
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
			label_->mutable_cpu_data()[0] = 1;
			label_->mutable_cpu_data()[3] = 1;
			label_->mutable_cpu_data()[6] = 1;
			label_->mutable_cpu_data()[7] = 1;
			bottom_.push_back(pred_);
			bottom_.push_back(label_);
			top_.push_back(loss_);
//			top_.push_back(prob_);
			layer_param_.mutable_softmax_param()->set_axis(-1);
		}

		LayerParameter layer_param_;

		Blob<Dtype>* pred_;
		Blob<Dtype>* label_;

		Blob<Dtype>* prob_;
		Blob<Dtype>* loss_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
	};
}

int main(int argc, char** argv){
	caffe::MultiLabelLossTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestBackward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestBackward(caffe::Caffe::GPU);
	return 0;
}