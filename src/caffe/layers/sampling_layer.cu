#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/random_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		caffe_gpu_rng_gaussian(count, mu_data, sigma_data, top_data);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		Dtype* mu_diff = bottom[0]->mutable_gpu_diff();
		Dtype* sigma_diff = bottom[1]->mutable_gpu_diff();
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		//TODO: add the gradient computation here
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SamplingLayer);
}
