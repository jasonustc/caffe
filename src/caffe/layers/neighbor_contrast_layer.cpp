#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){ 
		this->seq_len_ = this->layer_param_.recurrent_param().sequence_length();
		CHECK(bottom[0]->num() % seq_len_ == 0) << "number of samples should be" <<
			"dividable by sequence len";
		CHECK_GT(seq_len_, 0) << "sequence length must be positive";
	}

	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->num();
		const int n_seq = bottom[0]->count() / seq_len_;
		CHECK_GT(num, 1) << "input should have at least 2 frames";
		const num_contrast_out = n_seq * (seq_len_ - 1);
		top[0]->Reshape(num_contrast_out, bottom[0]->channels(), 
			bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->num();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const int n_seq = bottom[0]->count() / seq_len_;
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int offset = bottom[0]->shape(1);
		for (int n = 0; n < n_seq; n++){
			for (int i = 0; i < seq_len_ - 1; i++){
				caffe_sub(offset, bottom_data + offset, bottom_data, top_data);
				bottom_data += offset;
				top_data += offset;
			}
		}
	}

	template <typename Dtype>
	void NeighborContrastLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int n_seq = bottom[0]->count() / seq_len_;
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		const int offset = bottom[0]->shape(1);
		//set the difference of bottom to 0 first, then accumulate
		caffe_set(count, Dtype(0.), bottom_diff);
		if (propagate_down[0]){
			for (int n = 0; n < n_seq; n++){
				for (int i = 0; i < seq_len_ - 1; i++){
					caffe_add(offset, bottom_diff + offset, top_diff, bottom_diff + offset);
					caffe_sub(offset, bottom_diff, top_diff, bottom_diff);
					bottom_diff += offset;
					top_diff += offset;
				}
			}
		}
	}
}
