#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConstrainIPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.constrain_ip_param().num_output();
  bias_term_ = this->layer_param_.constrain_ip_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.constrain_ip_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.constrain_ip_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.constrain_ip_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  sum1_rate_ = this->layer_param_.constrain_ip_param().sum1_rate();
  monotonic_rate_ = this->layer_param_.constrain_ip_param().monotonic_rate();
}

template <typename Dtype>
void ConstrainIPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.constrain_ip_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ConstrainIPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	// hard constraints
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	for (int n = 0; n < N_; n++)
	{
		Dtype sum = 0;
		for (int k = K_-1; k >=0; k--)
		{
			int index = n*K_ + k;
			Dtype low_limit = (k==K_-1?0:weight_data[index+1]);
			if (weight_data[index] < low_limit)
				weight_data[index] = low_limit;
			sum += weight_data[index];
		}
		caffe_scal(K_, Dtype(1) / sum, &(weight_data[n*K_]));
	}



  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }


	/*for (int i = 0; i < K_; i++)
	{
		LOG(INFO) << bottom_data[i];
	}*/
	/*Dtype sum = 0;
	for (int i = 0; i < K_; i++)
	{
		LOG(INFO) << weight[i];
		sum += weight[i];
	}
	LOG(INFO) << sum;*/

}

template <typename Dtype>
void ConstrainIPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

	// Gradient w.r.t constraints on weights
	Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();
	const Dtype* weights_data = this->blobs_[0]->cpu_data();
	for (int i = 0; i < N_; i++)
	{
		if (sum1_rate_ != 0)
		{
			Dtype sum1_loss(1);
			for (int j = 0; j < K_; j++)
			{
				sum1_loss -= weights_data[i*K_ + j];
			}
			caffe_add_scalar(K_, Dtype(-1)*sum1_rate_*sum1_loss, &(weights_diff[i*K_]));
		}

		if (monotonic_rate_ != 0)
		{
			weights_diff[i*K_ + 0] += 
				monotonic_rate_*Dtype(1) / (weights_data[i*K_ + 0] - weights_data[i*K_ + 1]);
			for (int j = 1; j < K_ - 1; j++)
			{
				weights_diff[i*K_ + j] +=
					monotonic_rate_*(
					Dtype(1) / (weights_data[i*K_ + j] - weights_data[i*K_ + j + 1]) - 
					Dtype(1) / (weights_data[i*K_ + j - 1] - weights_data[i*K_ + j])
					);
			}
			weights_diff[i*K_ + K_-1] +=
				monotonic_rate_*(
				Dtype(1) / (weights_data[i*K_ + K_ - 1]) -
				Dtype(1) / (weights_data[i*K_ + K_ - 2] - weights_data[i*K_ + K_ - 1])
				);
		}
	}
		
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConstrainIPLayer);
#endif

INSTANTIATE_CLASS(ConstrainIPLayer);
REGISTER_LAYER_CLASS(ConstrainIP);

}  // namespace caffe
