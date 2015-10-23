#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->num();
		const int n_seq = bottom[0]->count() / seq_len_;
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int offset = bottom[0]->shape(1);
		for (int n = 0; n < n_seq; n++){
			for (int i = 0; i < seq_len_ - 1; i++){
				caffe_gpu_sub(offset, bottom_data + offset, bottom_data, top_data);
				bottom_data += offset;
				top_data += offset;
			}
		}
	}

	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int n_seq = bottom[0]->count() / seq_len_;
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int offset = bottom[0]->shape(1);
		//set the difference to 0 first, then accumulate
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		if (propagate_down[0]){
			for (int n = 0; n < n_seq; n++){
				for (int i = 0; i < seq_len_ - 1; i++){
					caffe_gpu_add(offset, bottom_diff + offset, top_diff, bottom_diff + offset);
					caffe_gpu_sub(offset, bottom_diff, top_diff, bottom_diff);
					bottom_diff += offset;
					top_diff += offset;
				}
			}
		}
	}
}
