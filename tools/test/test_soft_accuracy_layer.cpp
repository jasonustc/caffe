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
	class SoftAccuracyTest{
	public:
		SoftAccuracyTest() : pred_score_(new Blob<Dtype>()), true_score_(new Blob<Dtype>()),
			accuracy_(new Blob<Dtype>()), error_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestForward(Caffe::Brew caffe_mode){
			shared_ptr<Layer<Dtype>> layer(new SoftAccuracyLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			Caffe::set_mode(caffe_mode);
			layer->SetUp(bottom_, top_);
			//process the full sequence in a single batch
			layer->Forward(bottom_, top_);

			// Copy the inputs and outputs to reuse/check them later
			Dtype error_gap = this->layer_param_.accuracy_param().err_gap();
			const int count = pred_score_->count();
			Dtype sq_error = 0;
			Dtype accuracy = 0;
			const Dtype* pred_data = pred_score_->cpu_data();
			const Dtype* true_data = true_score_->cpu_data();
			for (int i = 0; i < count; i++){
				sq_error += (pred_data[i] - true_data[i]) * (pred_data[i] - true_data[i]);
				accuracy += abs(pred_data[i] - true_data[i]) < error_gap;
			}
			sq_error /= count;
			accuracy /= count;
			EXPECT_NEAR(sq_error, top_[1]->cpu_data()[0], 1e-4);
			EXPECT_NEAR(accuracy, top_[0]->cpu_data()[0], 1e-4);
		}

	protected:
		void SetUp(){
			layer_param_.mutable_accuracy_param()->set_err_gap(0.04);
			vector<int> score_shape(4, 1);
			score_shape[0] = 4;
			pred_score_->Reshape(score_shape);
			true_score_->Reshape(score_shape);
			pred_score_->mutable_cpu_data()[0] = 0.1;
			true_score_->mutable_cpu_data()[0] = 0.3;
			pred_score_->mutable_cpu_data()[2] = 0.15;
			true_score_->mutable_cpu_data()[2] = 0.16;
			bottom_.push_back(pred_score_);
			bottom_.push_back(true_score_);
			top_.push_back(accuracy_);
			top_.push_back(error_);
		}

		LayerParameter layer_param_;

		Blob<Dtype>* pred_score_;
		Blob<Dtype>* true_score_;

		Blob<Dtype>* accuracy_;
		Blob<Dtype>* error_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
	};
}

int main(int argc, char** argv){
	caffe::SoftAccuracyTest<float> test;
	test.TestForward(caffe::Caffe::CPU);
	return 0;
}