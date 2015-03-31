#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num_axes = bottom[0]->num_axes();
		const AxisPoolingParameter& AxisPooling_param = this->layer_param_.axis_pooling_param();
		pool_axis_ = bottom[0]->CanonicalAxisIndex(AxisPooling_param.axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape[pool_axis_] = 1;
		top[0]->Reshape(top_shape);
		num_pools_ = bottom[0]->count(0, pool_axis_);
		pool_input_size_ = bottom[0]->count(pool_axis_ + 1);
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();

		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		caffe_set(top[0]->count(), Dtype(0), top_data);
		for (int n = 0; n <num_pools_; ++n) {
			for (int i = 0; i < bottom_pool_axis; i++)
			{
				caffe_add( pool_input_size_,
					bottom_data + n * bottom_pool_axis * pool_input_size_+i*pool_input_size_,
					top_data + n*pool_input_size_, 
					top_data + n*pool_input_size_ 
					);
			}
		}
		caffe_scal(top[0]->count(), Dtype(1) / bottom_pool_axis, top_data);
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		if (!propagate_down[0])
			return;
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		for (int n = 0; n <num_pools_; ++n) {
			for (int i = 0; i < bottom_pool_axis; i++)
			{
				caffe_copy( pool_input_size_,
					top_diff + n*pool_input_size_, 
					bottom_diff + n * bottom_pool_axis * pool_input_size_+i*pool_input_size_
					);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(AxisPoolingLayer);
#endif

	INSTANTIATE_CLASS(AxisPoolingLayer);
	REGISTER_LAYER_CLASS(AxisPooling);

}  // namespace caffe
