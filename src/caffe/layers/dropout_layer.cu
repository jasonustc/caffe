#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const Dtype* mask, const Dtype threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    Dtype* mask =
        static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
	  if (this->drop_type_ == DropoutParameter_DROPTYPE_UNIFORM){
		caffe_gpu_rng_uniform(count, a_, b_, mask);
		DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, mask, a_, scale_, top_data);
	  }
	  else if (this->drop_type_ == DropoutParameter_DROPTYPE_GAUSSIAN){
		caffe_gpu_rng_gaussian(count, mu_, sigma_, mask);
		DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, mask, mu_ - sigma_, scale_, top_data);
	  }
	  else{
		caffe_gpu_rng_uniform(count, (Dtype)0., (Dtype)1., mask);
		DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, mask, threshold_, scale_, top_data);
	  }
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const Dtype* mask, const Dtype threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const Dtype* mask =
          static_cast<const Dtype*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
	  if (this->drop_type_ == DropoutParameter_DROPTYPE_GAUSSIAN){
		  DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS>>>(
			  count, top_diff, mask, mu_ - sigma_, scale_, bottom_diff);
	  }
	  else if (this->drop_type_ == DropoutParameter_DROPTYPE_UNIFORM){
		  DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS>>>(
			  count, top_diff, mask, a_, scale_, bottom_diff);
	  }
	  else{
		  // NOLINT_NEXT_LINE(whitespace/operators)
		  DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS>>>(
			  count, top_diff, mask, threshold_, scale_, bottom_diff);
	  }
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);


}  // namespace caffe
