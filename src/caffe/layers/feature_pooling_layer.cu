#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	
	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int pool_size = top[0]->count();
		caffe_gpu_memcpy(pool_size, bottom[0]->gpu_data(), top_data);
		for (int i = 1; i < bottom.size(); i++)
		{
			const Dtype *bottom_data = bottom[i]->gpu_data();
			caffe_gpu_add(pool_size, top_data, bottom_data, top_data);
		}
		caffe_gpu_scal(pool_size, (Dtype)1.0 / bottom.size(), top_data);
	}

	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* top_diff = top[0]->gpu_diff();
		const int pool_size = top[0]->count();
		for (int i = 0; i < bottom.size(); i++)
		{
			if (!propagate_down[i])
			{
				continue;
			}
			Dtype *bottom_diff = bottom[i]->mutable_gpu_diff();
			caffe_gpu_memcpy(pool_size, top_diff, bottom_diff);
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(FeaturePoolingLayer);
}// namespace caffe