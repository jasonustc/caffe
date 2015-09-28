#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/random_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*> &top){
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		//if there are no random transformations, we just copy bottom data to top blob
		//in test phase, we don't do any transformations
		if ((!scale_ && !rotation_ && !shift_) || this->phase_ == TEST){
			caffe_copy(count, bottom_data, top_data);
		}
		else{
			//get coordinate map matrix
			GetTransCoord();
			//Apply Imterpolation on bottom_data using tmat_[i] into top_data.
			InterpImageNN_gpu(bottom[0], coord_idx_->gpu_data(), top[0], INTERP_);
		}
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*> &bottom){
		const int count = top[0]->count();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		//Reset bottom diff.
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		//we must set the diff to zero before backpropagation
		if (propagate_down[0]){
			if (!scale_ && !shift_ && !rotation_){
				caffe_copy(count, top_diff, bottom_diff);
			}
			else{
				PropagateErrorNN_gpu(top[0], coord_idx_->gpu_data(), bottom[0], INTERP_);
			}
		}
	}

	//since the atomicAdd gpu function in transform only support float,
	//so we only register float functions here
	INSTANTIATE_LAYER_GPU_FUNCS_FLOAT_ONLY(RandomTransformLayer);
}
