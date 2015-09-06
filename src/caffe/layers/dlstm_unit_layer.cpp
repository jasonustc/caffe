#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"

namespace caffe{
	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return 1. / (1. - exp(x));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype x){
		return 2. * sigmoid(2. * x) - 1.;
	}

	//TODO: reverse the dim of bottom and top compared with LSTM layer
	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
			for(size_t i = 0; i < bottom.size(); i++){
				CHECK_EQ(bottom[i]->num_axes(), 3);
				CHECK_EQ(bottom[i]->shape(0), 1);
			}
		const int num_instances = bottom[0]->shape(1);
		hidden_dim_ = bottom[0]->shape(2);
		CHECK_EQ(bottom[1]->shape(2), 4 * hidden_dim_);
		CHECK_EQ(bottom[1]->shape(1), num_instances);
		CHECK_EQ(bottom[2]->shape(1), 1);
		CHECK_EQ(bottom[2]->shape(2), num_instances);
		top[0].ReshapeLike(*bottom[0]);
		top[1].ReshapeLike(*bottom[0]);
		X_acts_.ReshapeLike(*bottom[1]);
	} 

	//forward process of DLSTM is just like the backward process of LSTM
	//top: (#streams, #instances, ...)
	//bottom[0]: C_prev
	//bottom[1]: H_prev
	//bottom[1]: weight
	//bottom[2]: flush
	template<typename Dtype>
	void DLSTMUnitLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->shape(1);
		const int x_dim = hidden_dim_ * 4;
		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		const Dtype* flush = bottom[2]->cpu_data();
		Dtype* C = top[0]->cpu_data();
		Dtype* H = top[1]->cpu_data();
		for(int n = 0; n < num; n++){
			for(int d = 0; d < hidden_dim_; ++d){
				//it's the same for every sample in the mini-batch 
				const Dtype i = sigmoid(X[d]);
				const Dtype f = (*flush == 0) ? 0 :
					(*flush) * sigmoid(X[1 * hidden_dim_ + d]);
				const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
				const Dtype g = tanh(X[3 * hidden_dim_ + d]);
				const Dtype c_prev = C_prev[d];
				const Dtype c = C[d];
				const tanh_c = tanh(c);
			}
		}
	}
}