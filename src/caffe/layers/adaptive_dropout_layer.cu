#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidActivate(const int n, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = 1. / (1. + exp(-in[index]));
	}
}

template <typename Dtype>
__global__ void ReluActivate(const int n, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = in[index]> 0 ? in[index] : 0;
	}
}

template <typename Dtype>
inline void activate(const int n, const Dtype* in, Dtype* out,
	AdaptiveDropoutParameter_ActType act_type){
	switch (act_type){
	case caffe::AdaptiveDropoutParameter_ActType_SIGMOID:
		SigmoidActivate<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
		break;
	case caffe::AdaptiveDropoutParameter_ActType_RELU:
		ReluActivate<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
		break;
	default:
		LOG(FATAL) << "Unkown activate function.";
	}
}

template <typename Dtype>
__global__ void caffe_mult_and_add_scalar(const int n, const Dtype* in, Dtype* out,
	const Dtype alpha, const Dtype beta){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = alpha * in[index] + beta;
	}
}

template <typename Dtype>
void AdaptiveDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  Dtype* prob_weight_data = this->prob_weight_.mutable_gpu_data();
  Dtype* prob_data = this->prob_vec_.mutable_gpu_data();
  unsigned int *rand_vec_data = this->rand_vec_.mutable_gpu_data();
  const int count_weight = this->blobs_[0]->count();
  const int count_prob = this->prob_vec_.count();
  //compute prob_weight_data from weight_data
  //prob_weight_data = alpha_ * weight_data + beta_
  caffe_mult_and_add_scalar<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>
	  (count_weight, weight_data, prob_weight_data, alpha_, beta_);
  CUDA_POST_KERNEL_CHECK;
  //prob_data = alpha * op(bottom_data) * (prob_weight_data) + beta * prob_data
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, prob_weight_data, (Dtype)0., prob_data);
  activate(count_prob, prob_data, prob_data, this->prob_act_type_);
  //compute hidden units
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight_data, (Dtype)0., top_data);
  activate(top[0]->count(), top_data, top_data, this->hidden_act_type_);
  CUDA_POST_KERNEL_CHECK;
  if (this->phase_ == TRAIN){
	  caffe_gpu_rng_bernoulli<Dtype>(count_prob, prob_data, rand_vec_data);
	  caffe_gpu_mul_b<Dtype>(count_prob, top_data, rand_vec_data, top_data);
  }
  else{
	  caffe_gpu_mul<Dtype>(count_prob, top_data, prob_data, top_data);
  }
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void AdaptiveDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const int count_top = top[0]->count();
	Dtype* top_diff = top[0]->mutable_gpu_diff();
	if (this->phase_ == TRAIN){
		const unsigned int* rand_vec_data = this->rand_vec_.mutable_gpu_data();
		//top_diff = top_diff * rand_vec_data
		caffe_gpu_mul_b<Dtype>(count_top, top_diff, rand_vec_data, top_diff);
	}
	else{
		const Dtype* prob_vec_data = this->prob_vec_.mutable_gpu_data();
		caffe_gpu_mul<Dtype>(count_top, top_diff, prob_vec_data, top_diff);
	}
	if (this->param_propagate_down_[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		// Gradient with respect to weight
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
	}
	if (bias_term_ && this->param_propagate_down_[1]) {
		// Gradient with respect to bias
		caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
			bias_multiplier_.gpu_data(), (Dtype)0.,
			this->blobs_[1]->mutable_gpu_diff());
	}
	if (propagate_down[0]) {
		// Gradient with respect to bottom data
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
			bottom[0]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaptiveDropoutLayer);

}  // namespace caffe
