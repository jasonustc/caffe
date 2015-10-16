#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void CopyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		caffe_copy(count, bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
	}

	template <typename Dtype>
	void CopyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		if (propagate_down[0]){
			caffe_copy(count, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(CopyLayer);
#endif

	INSTANTIATE_CLASS(CopyLayer);
	REGISTER_LAYER_CLASS(Copy);
}