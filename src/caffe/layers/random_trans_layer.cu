/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: RandomTransformLayer(GPU)
*********************************************************************************/
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/random_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::GetTransCoord_gpu(){
		//here we use cpu to compute tranform matrix
		float* tmat_cpu_data = tmat_.mutable_cpu_data();
		if (rotation_){
			//randomly generate rotation angle
			caffe_rng_uniform(1, start_angle_, end_angle_, &curr_angle_);
			TMatFromParam(ROTATION, curr_angle_, curr_angle_, tmat_cpu_data);
		}
		if (scale_){
			caffe_rng_uniform(1, start_scale_, end_scale_, &curr_scale_);
			TMatFromParam(SCALE, curr_scale_, curr_scale_, tmat_cpu_data);
		}
		if (shift_){
			float shift_pixels_x = dx_prop_ * Width_;
			float shift_pixels_y = dy_prop_ * Height_;
			caffe_rng_uniform(1, -shift_pixels_x, shift_pixels_x, &curr_shift_x_);
			caffe_rng_uniform(1, -shift_pixels_y, shift_pixels_y, &curr_shift_y_);
			TMatFromParam(SHIFT, curr_shift_x_, curr_shift_y_, tmat_cpu_data);
		}
		//Canoincal size is set, so after finding the transformation,
		//crop or pad to that canonical size.
		//First find the coordinate matrix for this transformation
		//here we don't change the shape of the input 2D map
		GenCoordMatCrop_gpu(tmat_, Height_, Width_, original_coord_, coord_idx_, BORDER_, INTERP_);
	}

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
			GetTransCoord_gpu();
			//Apply Imterpolation on bottom_data using tmat_[i] into top_data.
			InterpImageNN_gpu(bottom[0], coord_idx_.gpu_data(), top[0], INTERP_);
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
		if (propagate_down[0]){
			if (!scale_ && !shift_ && !rotation_){
				caffe_copy(count, top_diff, bottom_diff);
			}
			else{
				BackPropagateErrorNN_gpu(top[0], coord_idx_.gpu_data(), bottom[0], INTERP_);
			}
		}
	}

	//since the atomicAdd gpu function in transform only support float,
	//so we only register float functions here
	INSTANTIATE_LAYER_GPU_FUNCS_FLOAT_ONLY(RandomTransformLayer);
}
