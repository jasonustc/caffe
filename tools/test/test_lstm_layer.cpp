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
	class LSTMTest{
	public:
		LSTMTest() : cont_(new Blob<Dtype>()), x_(new Blob<Dtype>()),
			h_enc_(new Blob<Dtype>()), c_T_(new Blob<Dtype>()),
			h_T_(new Blob<Dtype>()), unit_x_(new Blob<Dtype>()),
			cont_t_(new Blob<Dtype>()), unit_c_(new Blob<Dtype>()),
			top_c_t_(new Blob<Dtype>()), top_h_t_(new Blob<Dtype>()),
			x_t_(new Blob<Dtype>()), unit_cont_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestUnitSetUp(){
			LSTMUnitLayer<Dtype> layer(layer_param_);
			layer.SetUp(unit_bottom_, unit_top_);
			const int num_axes = unit_x_->num_axes();
			ASSERT_EQ(num_axes, this->top_h_t_->num_axes());
			ASSERT_EQ(num_axes, this->top_c_t_->num_axes());
			for (int i = 0; i < num_axes; i++){
				EXPECT_EQ(this->unit_c_->shape(i), this->top_h_t_->shape(i));
				EXPECT_EQ(this->unit_c_->shape(i), this->top_c_t_->shape(i));
			}
		}

		void TestUnitGradient(Caffe::Brew caffe_mode){
			Caffe::set_mode(caffe_mode);
			LSTMUnitLayer<Dtype> layer(layer_param_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			Dtype* flush_data = this->cont_t_->mutable_cpu_data();
			flush_data[0] = 1;
			checker.CheckGradientExhaustive(&layer, unit_bottom_, unit_top_, 0);
			checker.CheckGradientExhaustive(&layer, unit_bottom_, unit_top_, 1);
		}

		void TestUnitGradientZeroFlush(Caffe::Brew caffe_mode){
			Caffe::set_mode(caffe_mode);
			LSTMUnitLayer<Dtype> layer(layer_param_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			Dtype* flush_data = this->cont_t_->mutable_cpu_data();
			flush_data[0] = 0;
			checker.CheckGradientExhaustive(&layer, unit_bottom_, unit_top_, 0);
			checker.CheckGradientExhaustive(&layer, unit_bottom_, unit_top_, 1);
		}

		void TestForward(Caffe::Brew caffe_mode){
			shared_ptr<Layer<Dtype>> layer(new LSTMLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			Caffe::set_mode(caffe_mode);
			bottom_[0] = this->x_;
			bottom_[1] = this->cont_;
			layer->SetUp(bottom_, top_);
			//process the full sequence in a single batch
			LOG(INFO) << "Calling forward for a full sequence LSTM";
			layer->Forward(bottom_, top_);

			// Copy the inputs and outputs to reuse/check them later
			Blob<Dtype> bottom_copy(this->x_->shape());
			bottom_copy.CopyFrom(*(this->x_));
			Blob<Dtype> top_copy(this->h_enc_->shape());
			top_copy.CopyFrom(*(this->h_enc_));

			// Process the batch one timestep at a time
			// check that we get the same result
			layer_param_.mutable_recurrent_param()->set_sequence_length(1);
			layer.reset(new LSTMLayer<Dtype>(layer_param_));
			Caffe::set_random_seed(1701);
			bottom_[0] = this->x_t_;
			bottom_[1] = this->cont_t_;
			layer->SetUp(bottom_, top_);
			const int bottom_count = this->x_t_->count();
			const int top_count = this->h_enc_->count();
			const Dtype kEpsilon = 1e-5;
			for (int t = 0; t < 4; t++){
				caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
					this->x_t_->mutable_cpu_data());
				cont_t_->mutable_cpu_data()[0] = (t != 0 && t!= 2);
				LOG(INFO) << "Calling forward for LSTM timestep " << t;
				layer->Forward(bottom_, top_);
				//check output h_t data
				for (int i = 0; i < top_count; i++){
					ASSERT_LT(t * top_count + i, top_copy.count());
					EXPECT_NEAR(this->h_enc_->cpu_data()[i],
						top_copy.cpu_data()[t*top_count + i], kEpsilon)
						<< "t = " << t << "; i = " << i;
					if (t == 1 || t == 3){
						EXPECT_NEAR(this->h_T_->cpu_data()[i],
							top_copy.cpu_data()[t*top_count + i], kEpsilon);
					}
				}
			}

			// Process the batch one timestep at a time with all flush blobs set to 0.
			// Check that we get a different result, except in the first time step 
			// and third time step
			Caffe::set_random_seed(1701);
			layer.reset(new LSTMLayer<Dtype>(layer_param_));
			layer->SetUp(this->bottom_, this->top_);
			for (int t = 0; t < 4; t++){
				caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
					this->x_t_->mutable_cpu_data());
				cont_t_->mutable_cpu_data()[0] = 0;
				LOG(INFO) << "Calling forward for LSTM timestep " << t;
				layer->Forward(this->bottom_, this->top_);
				for (int i = 0; i < top_count; i++){
					if (t == 0 || t == 2){
						EXPECT_NEAR(this->h_enc_->cpu_data()[i],
							top_copy.cpu_data()[t*top_count + i], kEpsilon)
							<< "t = " << t << "; i = " << i;
					}
					else{
						EXPECT_NE(this->h_enc_->cpu_data()[i],
							top_copy.cpu_data()[t * top_count + i])
							<< "t = " << t << "; i = " << i;
					}
				}
			}
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new LSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 4);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 2);
			CHECK_EQ(top_[1]->shape(0), 2);
			CHECK_EQ(top_[1]->shape(1), 1);
			CHECK_EQ(top_[1]->shape(2), 2);
			CHECK_EQ(top_[2]->shape(0), 2);
			CHECK_EQ(top_[2]->shape(1), 1);
			CHECK_EQ(top_[2]->shape(2), 2);
		}

		void TestGradients(Caffe::Brew caffe_mode){
			layer_param_.mutable_recurrent_param()->set_decode(true);
			layer_param_.mutable_recurrent_param()->set_sequence_length(1);
			LSTMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(caffe_mode);
			bottom_[0] = x_;
			bottom_[1] = cont_;
			layer.SetUp(bottom_, top_);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			for (int i = 0; i < cont_->count(); i++){
				cont_->mutable_cpu_data()[i] = 0;
			}
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
		}

		void TestGradientsNonZeroFlush(Caffe::Brew caffe_mode){
			layer_param_.mutable_recurrent_param()->set_decode(true);
			layer_param_.mutable_recurrent_param()->set_sequence_length(2);
			LSTMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(caffe_mode);
			bottom_[0] = x_;
			bottom_[1] = cont_;
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			for (int i = 0; i < cont_->count(); i++){
				cont_->mutable_cpu_data()[i] = i > 0;
			}
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
		}

		
	protected:
		void SetUp(){
			layer_param_.mutable_recurrent_param()->set_num_output(2);
			layer_param_.mutable_recurrent_param()->set_decode(true);
			layer_param_.mutable_recurrent_param()->set_sequence_length(2);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_type("uniform");
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_min(0.);
			layer_param_.mutable_recurrent_param()->mutable_weight_filler()->set_max(1.);
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_recurrent_param()->mutable_bias_filler()->set_value(0.);
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			Caffe::set_random_seed(1);

			//for LSTM
			vector<int> cont_shape;
			cont_shape.push_back(4);
			cont_shape.push_back(1);
			cont_->Reshape(cont_shape);
			cont_shape[0] = 1;
			cont_t_->Reshape(cont_shape);
			vector<int> x_shape;
			x_shape.push_back(4);
			x_shape.push_back(1);
			x_shape.push_back(2);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			x_shape[0] = 1;
			x_t_->Reshape(x_shape);
			filler.Fill(x_);
			caffe_set(cont_->count(), Dtype(1), cont_->mutable_cpu_data());
			cont_->mutable_cpu_data()[0] = 0;
			cont_->mutable_cpu_data()[2] = 0;
			bottom_.push_back(x_);
			bottom_.push_back(cont_);
			top_.push_back(h_enc_);
			top_.push_back(h_T_);
			top_.push_back(c_T_);

			//for LSTM Unit
			vector<int> unit_cont_shape(3, 1);
			unit_cont_->Reshape(unit_cont_shape);
			vector<int> unit_x_shape(3, 1);
			unit_x_shape[2] = 8;
			unit_x_->Reshape(unit_x_shape);
			filler.Fill(unit_x_);
			unit_x_shape[2] = 2;
			unit_c_->Reshape(unit_x_shape);
			filler.Fill(unit_c_);
			unit_bottom_.push_back(unit_c_);
			unit_bottom_.push_back(unit_x_);
			unit_bottom_.push_back(unit_cont_);
			unit_top_.push_back(top_h_t_);
			unit_top_.push_back(top_c_t_);
		}

		LayerParameter layer_param_;

		Blob<Dtype>* cont_;
		Blob<Dtype>* x_;
		Blob<Dtype>* x_t_;
		Blob<Dtype>* cont_t_;

		Blob<Dtype>* unit_cont_;
		Blob<Dtype>* unit_x_;
		Blob<Dtype>* unit_c_;

		Blob<Dtype>* top_h_t_;
		Blob<Dtype>* top_c_t_;

		Blob<Dtype>* h_enc_;

		Blob<Dtype>* h_T_;
		Blob<Dtype>* c_T_;


		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<Blob<Dtype>*> unit_bottom_;
		vector<Blob<Dtype>*> unit_top_;

	};
}

int main(int argc, char** argv){
	caffe::LSTMTest<float> test;
	test.TestUnitSetUp();
	test.TestSetUp();
	test.TestUnitGradient(caffe::Caffe::CPU);
	test.TestUnitGradientZeroFlush(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestGradientsNonZeroFlush(caffe::Caffe::CPU);
	test.TestUnitGradient(caffe::Caffe::GPU);
	test.TestUnitGradientZeroFlush(caffe::Caffe::GPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::GPU);
	test.TestGradientsNonZeroFlush(caffe::Caffe::GPU);
	return 0;
}