
/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/1/1
** desc: SimMergeLayer(GPU), merge similar feature maps and re-initialize similar
**       weights to learn more independent feature maps
*********************************************************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
namespace caffe{

	//TODO: maybe this operation will be very time consuming, we 
	// need to figure out a more efficient way
	template <typename Dtype>
	void SimMergeLayer<Dtype>::update_sim_matrix_gpu(const vector<Blob<Dtype>*>& top){
		const int num = top[0]->count(0, axis_);
		const int channel = top[0]->shape(axis_);
		const int dim = top[0]->count(axis_ + 1);
		const Dtype* top_data = top[0]->gpu_data();
		Dtype* temp_data = this->temp_avg_map_.mutable_gpu_data();
		//to save memory, put history similarity in diff
		//and current similarity in data
		Dtype* curr_sim_data = this->sim_.mutable_gpu_diff();
		Dtype* his_sim_data = this->sim_.mutable_gpu_data();
		int count = 0;
		//average through batch
		for (int i = 0; i < num; i++){
			caffe_gpu_add(channel * dim, top_data + i * channel * dim, temp_data, temp_data);
		}
		caffe_gpu_scal<Dtype>(channel * dim, Dtype(1.) / (Dtype)num, temp_data);
		for (int i = 0; i < channel; i++){
			for (int j = i + 1; j < channel; j++){
				Dtype inner_prod;
				caffe_gpu_dot(dim, temp_data + i * dim,
					temp_data + j * dim, &inner_prod);
				Dtype sumsq_a;
				caffe_gpu_dot(dim, temp_data + i * dim,
					temp_data + i * dim, &sumsq_a);
				Dtype sumsq_b;
				caffe_gpu_dot(dim, temp_data + j * dim,
					temp_data + j * dim, &sumsq_b);
				Dtype sim = inner_prod / (sqrt(sumsq_a * sumsq_b + FLT_MIN));
				//					curr_sim_data[count] = sim;
				caffe_gpu_set(1, sim, curr_sim_data + count);
				count++;
			}
		}
		//update history similarity with current similarity
		const Dtype curr_iter = 1 + this->curr_iter_;
		caffe_gpu_axpby(count, (Dtype)1. / curr_iter, curr_sim_data,
			(Dtype)this->curr_iter_ / curr_iter, his_sim_data);
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::merge_two_feature_maps_gpu(const vector<Blob<Dtype>*>& top,
		const int m, const int n, const Dtype sim){
		const int num = top[0]->num();
		const int offset = top[0]->count(this->axis_);
		const int feat_dim = top[0]->count(this->axis_ + 1);
		for (int i = 0; i < num; i++){
			Dtype* map_m_data = top[0]->mutable_gpu_data() + i * offset + m * feat_dim;
			const Dtype* map_n_data = top[0]->mutable_gpu_data() + i * offset + n * feat_dim;
			const Dtype denom = 1. + sim;
			caffe_gpu_axpby(feat_dim, Dtype(sim) / denom, map_n_data, 
				Dtype(1.) / denom, map_m_data);
		}
	}

	//Reset weights/bias data to random initialized and 
	//reseet weight/bias diff to 0.
	//Here we assume that weight is in [num_out, ...] format
	//and bias is in [num_out, ...] format
	template <typename Dtype>
	void SimMergeLayer<Dtype>::refresh_weight_gpu(const int j){
		Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		const int num = this->blobs_[0]->num();
		const int dim = this->blobs_[0]->count() / num;
		Dtype* weight_offset_data = weight_data + this->blobs_[0]->offset(j);
		Dtype* weight_offset_diff = weight_diff + this->blobs_[0]->offset(j);
		this->weight_filler_->Fill_gpu(weight_offset_data, dim);
		caffe_gpu_set(dim, (Dtype)0., weight_offset_diff);
		Dtype* data_1 = this->blobs_[0]->mutable_cpu_data();
		Dtype* data_2 = this->blobs_[0]->mutable_gpu_data();
		if (bias_term_){
			const int bias_dim = this->blobs_[1]->count(1);
			Dtype* bias_offset_data = this->blobs_[1]->mutable_gpu_data() + 
				this->blobs_[1]->offset(j);
			Dtype* bias_offset_diff = this->blobs_[1]->mutable_gpu_diff() + 
				this->blobs_[1]->offset(j);
			this->bias_filler_->Fill_gpu(bias_offset_data, bias_dim);
			caffe_gpu_set(bias_dim, (Dtype)0., bias_offset_diff);
		}
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::merge_sim_feature_maps_gpu(const vector<Blob<Dtype>*>& top){
		const int channel = top[0]->shape(this->axis_);
		const Dtype* sim_data = this->sim_.cpu_data();
		int index = 0;
		vector<int> merged_map_ids;
		for (int i = 0; i < channel; i++){
			//if map i has already been merged, we just skip it
			if (std::find(merged_map_ids.begin(), merged_map_ids.end(), i)
				!= merged_map_ids.end()){
				continue;
			}
			for (int j = i + 1; j < channel; j++){
				if (sim_data[index] > this->threshold_){
					merged_map_ids.push_back(j);
					this->merge_two_feature_maps_gpu(top, i, j, sim_data[index]);
					if (weight_term_){
						//re-initialize the weight
						this->refresh_weight_gpu(j);
					}
				}
				index++;
			}
		}
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (weight_term_){
			DCHECK(this->blobs_[0]->count()) << "Please check if the name of weight "
				<< "parameter is shared by other layer";
		}
		if (bias_term_){
			DCHECK(this->blobs_[1]->count()) << "Please check if the name of bias "
				<< "parameter is shared by other layer";
		}
		this->update_sim_matrix_gpu(bottom);
		this->curr_iter_++;
		if (this->curr_iter_ % this->iter_ == 0){
			//reset number of iterations, 
			//so as to reset similarity matrix to all 0s
			this->curr_iter_ = 0;
			this->merge_sim_feature_maps_gpu(bottom);
		}
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		//currently, we have nothing to do
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SimMergeLayer);
}
