#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/sequence_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();

		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
		for (int n = 0; n <num_pools_; ++n) {
			for (int i = 0; i < bottom_pool_axis; i++)
			{
				caffe_gpu_add(pool_input_size_,
					bottom_data + n * bottom_pool_axis * pool_input_size_ + i*pool_input_size_,
					top_data + n*pool_input_size_,
					top_data + n*pool_input_size_
					);
			}
		}
		caffe_gpu_scal(top[0]->count(), Dtype(1) / bottom_pool_axis, top_data);
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		if (!propagate_down[0])
			return;
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		for (int n = 0; n <num_pools_; ++n) {
			for (int i = 0; i < bottom_pool_axis; i++)
			{
				caffe_copy(pool_input_size_,
					top_diff + n*pool_input_size_,
					bottom_diff + n * bottom_pool_axis * pool_input_size_ + i*pool_input_size_
					);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(AxisPoolingLayer);
}  // namespace caffe
