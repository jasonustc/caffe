#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
	__global__ void MultiLabelLossForwardGPU(const int nthreads,
		const Dtype* input_data, const Dtype* prob_data, const Dtype* label, Dtype* loss,
		caffe::MultiLabelLossParameter_ProbType prob_type){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int label_value = static_cast<int>(label[index]);
			if (prob_type == caffe::MultiLabelLossParameter_ProbType_SIGMOID){
				if (label_value != 0){
					loss[index] = log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0)))
						- (input_data[index] * ((label_value > 0) - (input_data[index] >= 0)));
				}
				else{
					loss[index] = 0;
				}
			}
			else{
				if (label_value == 1){
					loss[index] = -log(max(prob_data[index], Dtype(FLT_MIN)));
				}
				else{
					loss[index] = 0;
				}
			}
		}
	}

	//Here we just normalize the loss by batch_size
	//TODO: add the choice of nomalizing by counts
	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const Dtype* prob_data = prob_.gpu_data();
		const Dtype* label = bottom[1]->gpu_data();
		const Dtype* input_data = bottom[0]->gpu_data();
		//since this memory is not used for anything util it is
		//overwritten on the backward pass, we use it here to 
		//avoid having to allocate new GPU memory to accumulate 
		//intermediate results in the kernel.
		Dtype* loss_data = bottom[0]->mutable_gpu_diff();
		MultiLabelLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, input_data, prob_data, 
			label, loss_data, this->prob_type_);
		Dtype loss;
		//because -log(prob) is always positive, so we can use asum here
		//to get the summation of the loss
		caffe_gpu_asum(count, loss_data, &loss);
		top[0]->mutable_cpu_data()[0] = loss / num;
		if (top.size() > 1){
			top[1]->ShareData(prob_);
		}
	}

	//TODO: the gradient for multi-label softmax is not right here
	template <typename Dtype>
	__global__ void MultiLabelLossBackwardGPU(const int count,
		const Dtype* prob_data, const Dtype* label, Dtype* bottom_diff,
		caffe::MultiLabelLossParameter_ProbType prob_type){
		CUDA_KERNEL_LOOP(index, count){
			const int label_value = static_cast<int>(label[index]);
			if (prob_type == caffe::MultiLabelLossParameter_ProbType_SIGMOID){
				if (label_value != 0){
					bottom_diff[index] = prob_data[index] - (label_value > 0);
				}
				else{
					bottom_diff[index] = 0;
				}
			}
			else{
				if (label_value == 1){
					bottom_diff[index] = prob_data[index] - 1;
				}
				else{
					bottom_diff[index] = prob_data[index];
				}
			}
		}
	}

	template <typename Dtype>
	void MultiLabelLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[1]){
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label";
		}
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		if (propagate_down[0]){
			const Dtype* prob_data = prob_.gpu_data();
			const Dtype* input_data = bottom[0]->gpu_data();
			const Dtype* label = bottom[1]->gpu_data();
			MultiLabelLossBackwardGPU<Dtype><< <CAFFE_GET_BLOCKS(count), 
				CAFFE_CUDA_NUM_THREADS>> >(count, prob_data, label, bottom_diff, this->prob_type_);
		}
		const int num = bottom[0]->num();
		const Dtype loss_weight = top[0]->cpu_diff()[0] / num;
		caffe_gpu_scal(count, loss_weight, bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MultiLabelLossLayer);

}//namespace caffe
