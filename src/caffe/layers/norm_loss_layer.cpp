#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

	template<typename Dtype>
	void NormLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		vector<int> top_shape(0);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void NormLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int dim = count / num;
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* loss = top[0]->mutable_cpu_data();
		switch (this->layer_param_.norm_loss_param().norm_type()){
		case NormLossParameter_NormType_L1:
			loss[0] = caffe_cpu_asum(count, bottom_data) / num;
			break;
		case NormLossParameter_NormType_L2:
			loss[0] = caffe_cpu_dot(count, bottom_data, bottom_data) / num / 2;
			break;
		default:
			LOG(FATAL) << "Unkown Norm Type";
		}
		bottom[0]->ToTxt("bottom");
		top[0]->ToTxt("top");
	}

	template <typename Dtype>
	void NormLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const int count = bottom[0]->count();
			const int num = bottom[0]->num();
			const int dim = count / num;
			const Dtype loss_weight = top[0]->cpu_diff()[0];
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype alpha = loss_weight / num;
			switch (this->layer_param_.norm_loss_param().norm_type()){
			case NormLossParameter_NormType_L1:
				caffe_cpu_sign(count, bottom_data, bottom_diff);
				caffe_scal(count, loss_weight / num, bottom_diff);
				break;
			case NormLossParameter_NormType_L2:
				caffe_cpu_axpby(count, alpha, bottom_data, Dtype(0), bottom_diff);
				break;
			default:
				LOG(FATAL) << "Unkown Norm Type.";
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(NormLossLayer)
#endif

	INSTANTIATE_CLASS(NormLossLayer);
	REGISTER_LAYER_CLASS(NormLoss);
}