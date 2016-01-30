#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe{
	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom[0]->num(), bottom[1]->num()) <<
			"MultiLabelLoss layer inputs must have the same num.";
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) <<
			"The predict score must be equal to the number of classes";
		this->prob_type_ = this->layer_param_.multi_label_loss_param().prob_type();
		sigmoid_bottom_vec_.clear();
		sigmoid_bottom_vec_.push_back(bottom[0]);
		sigmoid_top_vec_.clear();
		sigmoid_top_vec_.push_back(&prob_);
		LayerParameter sigmoid_param(this->layer_param_);
		switch (prob_type_){
		case MultiLabelLossParameter_ProbType_SIGMOID:
			sigmoid_param.set_type("Sigmoid");
			break;
		case MultiLabelLossParameter_ProbType_SOFTMAX:
			sigmoid_param.set_type("Softmax");
			break;
		default:
			LOG(FATAL) << "Unknown probability type";
		}
		sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
		sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
	}

	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::Reshape(bottom, top);
		if (top.size() > 1){
			//output sigmoid values
			top[1]->ReshapeLike(prob_);
			top[1]->ShareData(prob_);
		}
	}

	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//because Forward_cpu is protected funtion, we cannot call it
		//outside the layer
		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		// negative log likelihood
		const int count = bottom[0]->count();
		// Stable version of loss computation from input data
		const Dtype* input_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* prob_data = prob_.cpu_data();
		Dtype loss = 0;
		for (int i = 0; i < count; ++i){
			const int label_value = static_cast<int>(label[i]);
			//update the loss only if the target[i] is not 0
			if (label_value != 0){
				if (this->prob_type_ == MultiLabelLossParameter_ProbType_SIGMOID){
					/*
					 * target[i] == 1(positive):
					 * loss = log(1+exp{-x}), x >= 0;
					 *        -x + log(1 + exp{x}), x < 0.
					 * target[i] == -1(negative):
					 * loss = log(1+exp{x}), x < 0;
					 *        x - log(1 + exp{-x}), x >= 0.
					 */
					loss -= input_data[i] * ((label_value > 0) - (input_data[i] >= 0)) -
						log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
				}
				else{
					loss -= log(std::max(prob_data[i], Dtype(FLT_MIN)));
				}
			}
		}
		//average by batch size
		top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		if (top.size() > 1){
			top[1]->ShareData(prob_);
		}
	}

	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[1]){
			LOG(FATAL) << this->type()
				<< " Layer can not backpropagate to label inputs.";
		}
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]){
			const Dtype* prob_data = prob_.cpu_data();
			const Dtype* label = bottom[1]->cpu_data();
			const int count = bottom[0]->count();
			if (this->prob_type_ == MultiLabelLossParameter_ProbType_SIGMOID){
				for (int i = 0; i < count; ++i){
					const int label_value = static_cast<int>(label[i]);
					if (label_value != 0){
						bottom_diff[i] = prob_data[i] - (label_value > 0);
					}
					else{
						//0: ignore, no gradient updates
						bottom_diff[i] = 0;
					}
				}
			}
			else{
				caffe_copy(prob_.count(), prob_data, bottom_diff);
				for (int i = 0; i < count; i++){
					const int label_value = static_cast<int>(label[i]);
					if (label_value == 1){
						bottom_diff[i] -= 1;
					}
				}
			}
		}
		//scale gradient
		Dtype loss_weight = top[0]->cpu_diff()[0] / bottom[0]->num();
		caffe_scal(prob_.count(), loss_weight, bottom_diff);
	}

#ifdef CPU_ONLY
	STUB_GPU(MultiLabelLossLayer);
#endif

	INSTANTIATE_CLASS(MultiLabelLossLayer);
	REGISTER_LAYER_CLASS(MultiLabelLoss);
}//namespace caffe
