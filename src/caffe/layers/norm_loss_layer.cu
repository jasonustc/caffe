#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

	template <typename Dtype>
	void NormLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype scale = 1. / Dtype(num);
		switch (this->layer_param_.norm_loss_param().norm_type()){
		case NormLossParameter_NormType_L1:
			caffe_gpu_asum(count, bottom_data, top_data);
			caffe_scal(top[0]->count(), scale, top_data);
			break;
		case NormLossParameter_NormType_L2:
			caffe_gpu_dot(count, bottom_data, bottom_data, top_data);
			caffe_scal(top[0]->count(), scale * Dtype(0.5), top_data);
			break;
		default:
			LOG(FATAL) << "Unkown Norm Type.";
		}
	}

	template <typename Dtype>
	void NormLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int dim = count / num;
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype loss_weight = top[0]->gpu_diff()[0];
		Dtype alpha = loss_weight / num;
		switch (this->norm_type_)
		{
		case NormLossParameter_NormType_L1:
			caffe_gpu_sign(count, bottom_data, bottom_diff);
			caffe_scal(count, loss_weight / num, bottom_diff);
			break;
		case NormLossParameter_NormType_L2:
			caffe_gpu_axpby(count, alpha, bottom_data, Dtype(0), bottom_diff);
			break;
		default:
			LOG(FATAL) << "Unkown Norm Type.";
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NormLossLayer);
}

