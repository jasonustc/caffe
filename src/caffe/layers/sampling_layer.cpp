#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/random_layers.hpp"


namespace caffe{
	template <typename Dtype>
	void SamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Layer<Dtype>::LayerSetUp(bottom, top);
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << "The shape of mu and sigma \
			should be the same";
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->cpu_data();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		const int count = bottom[0]->count();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_rng_gaussian(count, mu_data, sigma_data, top_data);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top){
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* mu_diff = bottom[0]->mutable_cpu_diff();
		Dtype* sigma_diff = bottom[1]->mutable_cpu_diff();
		//TODO: add the gradient computation here
	}

	INSTANTIATE_CLASS(SamplingLayer);
	REGISTER_LAYER_CLASS(Sampling);
}