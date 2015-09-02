#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	
	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
	}

	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		const int num_axes = bottom[0]->num_axes();

		//Initialize with the first blob
		vector<int> top_shape = bottom[0]->shape();
		//int bootom_count_sum = bottom[0]->count();
		for (int i = 1; i < bottom.size(); i++)
		{
			CHECK_EQ(num_axes, bottom[i]->num_axes())
				<< "All inputs must have the same #axes.";
			for (int j = 0; j < num_axes; j++)
			{
				CHECK_EQ(top_shape[j], bottom[i]->shape(j))
					<< "all inputs must have the same shape.";
			}
		}
		top[0]->Reshape(top_shape);
	}

	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int pool_size = top[0]->count();
		caffe_copy(pool_size, bottom[0]->cpu_data(), top_data);
		for (int i = 1; i < bottom.size(); i++)
		{
			const Dtype *bottom_data = bottom[i]->cpu_data();
			caffe_add(pool_size, top_data, bottom_data, top_data);
		}
		caffe_scal(pool_size, (Dtype)1.0 / bottom.size(), top_data);
	}

	template<typename Dtype>
	void FeaturePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		const int pool_size = top[0]->count();
		for (int i = 0; i < bottom.size(); i++)
		{
			if (!propagate_down[i])
			{
				continue;
			}
			Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
			caffe_copy(pool_size, top_diff, bottom_diff);
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(FeaturePoolingLayer);
#endif

	INSTANTIATE_CLASS(FeaturePoolingLayer);
	REGISTER_LAYER_CLASS(FeaturePooling);
}// namespace caffe