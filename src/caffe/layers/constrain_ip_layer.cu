#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConstrainIPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// hard constraints
	Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
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
		caffe_gpu_scal(K_, Dtype(1) / sum, &(weight_data[n*K_]));
	}
	
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }

}

template <typename Dtype>
void ConstrainIPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

		//// Gradient w.r.t constraints on weights
		//Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();
		//const Dtype* weights_data = this->blobs_[0]->gpu_data();
		//for (int i = 0; i < N_; i++)
		//{
		//	if (sum1_rate_ != 0)
		//	{
		//		Dtype sum1_loss(1);
		//		for (int j = 0; j < K_; j++)
		//		{
		//			sum1_loss -= weights_data[i*K_ + j];
		//		}
		//		caffe_gpu_add_scalar(K_, Dtype(-1)*sum1_rate_*sum1_loss, &(weights_diff[i*K_]));
		//	}

		//	if (monotonic_rate_ != 0)
		//	{
		//		weights_diff[i*K_ + 0] +=
		//			monotonic_rate_*Dtype(1) / (weights_data[i*K_ + 0] - weights_data[i*K_ + 1]);
		//		for (int j = 1; j < K_ - 1; j++)
		//		{
		//			weights_diff[i*K_ + j] +=
		//				monotonic_rate_*(
		//				Dtype(1) / (weights_data[i*K_ + j] - weights_data[i*K_ + j + 1]) -
		//				Dtype(1) / (weights_data[i*K_ + j - 1] - weights_data[i*K_ + j])
		//				);
		//		}
		//		weights_diff[i*K_ + K_ - 1] +=
		//			monotonic_rate_*(
		//			Dtype(1) / (weights_data[i*K_ + K_ - 1]) -
		//			Dtype(1) / (weights_data[i*K_ + K_ - 2] - weights_data[i*K_ + K_ - 1])
		//			);
		//	}
		//}
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConstrainIPLayer);

}  // namespace caffe
