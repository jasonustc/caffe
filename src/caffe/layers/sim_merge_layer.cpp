/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/1/1
** desc: SimMergeLayer(CPU), merge similar feature maps and re-initialize similar
**       weights to learn more independent feature maps
*********************************************************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
void SimMergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	//specified operations for similarity based feature map merge
	iter_ = this->layer_param_.sim_merge_param().iter();
	threshold_ = this->layer_param_.sim_merge_param().threshold();
	axis_ = this->layer_param_.sim_merge_param().axis();
	curr_iter_ = 0;
	bias_term_ = this->layer_param_.sim_merge_param().has_bias_filler();
	weight_term_ = this->layer_param_.sim_merge_param().has_weight_filler();
	const int n = this->layer_param_.param_size();
	CHECK_EQ(weight_term_ + bias_term_, this->layer_param_.param_size())
		<< "Number of fillers must be equal to number of shared parameters";
	//weight filler and bias filler
	if (weight_term_){
		CHECK(this->layer_param_.sim_merge_param().has_weight_shape())
			<< "specified shape of weight should be provided";
		this->weight_filler_.reset(GetFiller<Dtype>(this->layer_param_.sim_merge_param().weight_filler()));
	}
	if (bias_term_){
		CHECK(this->layer_param_.sim_merge_param().has_bias_shape())
			<< "specified shape of bias should be provided";
		this->bias_filler_.reset(GetFiller<Dtype>(this->layer_param_.sim_merge_param().bias_filler()));
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	//count of maps
	const int count = bottom[0]->shape(axis_);
	CHECK_GT(count, 1) << "Only more than 1 features maps need to be merged";
	this->sim_.Reshape(1, 1, 1, count * (count - 1) / 2);
	this->temp_avg_map_.Reshape(1, channels, height, width);
	this->blobs_.resize(bias_term_ + weight_term_);
	//maybe here we need to allocate memory first
	//then share from bottom layer parameter in AppendParam()
	if (weight_term_){
		const BlobShape& weight_shape = this->layer_param_.sim_merge_param().weight_shape();
		CHECK_EQ(count, weight_shape.dim(0)) << "number of feature maps should be equal with "
			<< "number of weights";
		vector<int> shape_weight;
		for (int i = 0; i < weight_shape.dim_size(); i++){
			shape_weight.push_back(weight_shape.dim(i));
		}
		this->blobs_[0].reset(new Blob<Dtype>(shape_weight));
		LOG(INFO) << "weight shape: " << this->blobs_[0]->shape_string();
	}
	if (bias_term_){
		const BlobShape& bias_shape = this->layer_param_.sim_merge_param().bias_shape();
		CHECK_EQ(count, bias_shape.dim(0)) << "number of feature maps should be equal with "
			<< "number of biases";
		vector<int> shape_bias;
		for (int i = 0; i < bias_shape.dim_size(); i++){
			shape_bias.push_back(bias_shape.dim(i));
		}
		this->blobs_[1].reset(new Blob<Dtype>(shape_bias));
		LOG(INFO) << "bias shape: " << this->blobs_[1]->shape_string();
	}
}

//TODO: maybe this operation will be very time consuming, we 
// need to figure out a more efficient way
template <typename Dtype>
void SimMergeLayer<Dtype>::update_sim_matrix_cpu(const vector<Blob<Dtype>*>& top){
	const int num = top[0]->count(0, axis_);
	const int channel = top[0]->shape(axis_);
	const int dim = top[0]->count(axis_ + 1);
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* temp_data = this->temp_avg_map_.mutable_cpu_data();
	//to save memory, put history similarity in data
	//and current similarity in diff
	Dtype* curr_sim_data = this->sim_.mutable_cpu_diff();
	Dtype* his_sim_data = this->sim_.mutable_cpu_data();
	//average by batch_size
	for (int i = 0; i < num; i++){
		caffe_add(channel * dim, top_data + i * channel * dim, temp_data, temp_data);
	}
	caffe_scal<Dtype>(channel * dim, Dtype(1.) / (Dtype)num, temp_data);
	int count = 0;
	for (int i = 0; i < channel; i++){
		for (int j = i + 1; j < channel; j++){
			Dtype inner_prod = caffe_cpu_dot(dim, temp_data + i * dim,
				temp_data + j * dim);
			Dtype sumsq_a = caffe_cpu_dot(dim, temp_data + i * dim,
				temp_data + i * dim);
			Dtype sumsq_b = caffe_cpu_dot(dim, temp_data + j * dim,
				temp_data + j * dim);
			Dtype sim = inner_prod / (sqrt(sumsq_a * sumsq_b + FLT_MIN));
			curr_sim_data[count] = sim;
			count++;
		}
	}
	this->sim_.ToTxt("sim_before_merge", true);
	//update history similarity with current similarity
	const Dtype curr_iter = 1 + this->curr_iter_;
	caffe_cpu_axpby(count, (Dtype)1. / (Dtype)curr_iter, curr_sim_data, 
		(Dtype)this->curr_iter_ / (Dtype)curr_iter, his_sim_data);
	this->sim_.ToTxt("sim_after_merge", true);
}

//merge feature map n to feature map m
template <typename Dtype>
void SimMergeLayer<Dtype>::merge_two_feature_maps_cpu(const vector<Blob<Dtype>*>& top,
	const int m, const int n, const Dtype sim){
	const int num = top[0]->count(0, this->axis_);
	const int offset = top[0]->count(this->axis_);
	const int feat_dim = top[0]->count(this->axis_ + 1);
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype denom = 1 + sim;
	for (int i = 0; i < num; i++){
		Dtype* map_m_data = top_data + i * offset + m * feat_dim;
		const Dtype* map_n_data = top_data + i * offset + n * feat_dim;
		caffe_cpu_axpby(feat_dim, Dtype(sim) / denom, map_n_data, 
			Dtype(1.) / denom, map_m_data);
	}
}

//Reset weights/bias data to random initialized and 
//reseet weight/bias diff to 0.
//Here we assume that weight is in [num_out, ...] format
//and bias is in [num_out, ...] format
template <typename Dtype>
void SimMergeLayer<Dtype>::refresh_weight_cpu(const int j){
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	const int num = this->blobs_[0]->num();
	const int dim = this->blobs_[0]->count() / num;
	Dtype* weight_offset_data = weight_data + this->blobs_[0]->offset(j);
	Dtype* weight_offset_diff = weight_diff + this->blobs_[0]->offset(j);
	this->weight_filler_->Fill(weight_offset_data, dim);
	caffe_set(dim, (Dtype)0., weight_offset_diff);
	if (bias_term_){
		const int bias_dim = this->blobs_[1]->count(1);
		Dtype* bias_offset_data = this->blobs_[1]->mutable_cpu_data() + 
			this->blobs_[1]->offset(j);
		Dtype* bias_offset_diff = this->blobs_[1]->mutable_cpu_diff() + 
			this->blobs_[1]->offset(j);
		this->bias_filler_->Fill(bias_offset_data, bias_dim);
		caffe_set<Dtype>(bias_dim, (Dtype)0., bias_offset_diff);
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::merge_sim_feature_maps_cpu(const vector<Blob<Dtype>*>& top){
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
				this->merge_two_feature_maps_cpu(top, i, j, sim_data[index]);
				if (weight_term_){
					//re-initialize the weight
					this->refresh_weight_cpu(j);
				}
			}
			index++;
		}
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->update_sim_matrix_cpu(bottom);
	this->curr_iter_++;
	if (this->curr_iter_ % this->iter_ == 0){
		//reset number of iterations, 
		//so as to reset similarity matrix to all 0s
		this->curr_iter_ = 0;
		this->merge_sim_feature_maps_cpu(bottom);
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//currently, we have nothing to do
}

#ifdef CPU_ONLY
STUB_GPU(SimMergeLayer);
#endif

INSTANTIATE_CLASS(SimMergeLayer);
REGISTER_LAYER_CLASS(SimMerge);
}