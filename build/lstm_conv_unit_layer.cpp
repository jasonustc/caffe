#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"

namespace caffe{
	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return (Dtype)1. / ((Dtype)1. + exp(x));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype x){
		return (Dtype)2. * sigmoid(Dtype(2.) * x) - 1.;
	 }

	template <typename Dtype>
	inline Dtype relu(Dtype x){
		return x > 0 ? x : Dtype(0);
	}

	//here just follow the LSTM work to make equal size of h and c during
	//every time step
	//TODO: maybe we can try varies the size of feature maps(h and c) 
	//like in CNN architectures
	template <typename Dtype>
	void LSTMConvUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//we must change the features into 2-D map
		//1 x #channels x width x height
		for (int i = 0; i < bottom.size(); i++){
			CHECK_EQ(4, bottom[i]->num_axes());
			CHECK_EQ(1, bottom[i]->shape(0));
		}
		const int num_channels = bottom[0]->shape(1);
		hidden_dim_ = bottom[0]->count(2);
		//what does bottom[2]->shape(2) mean?
		CHECK_EQ(num_channels, bottom[1]->shape(1));
		//dim of cont?
		CHECK_EQ(num_channels, bottom[2]->shape(2));
		CHECK_EQ(1, bottom[2]->shape(1));
		CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
		//c
		top[0]->ReshapeLike(*bottom[0]);
		//h
		top[1]->ReshapeLike(*bottom[0]);
		//x and 3 gates(f, o, i)
		X_acts_.ReshapeLike(*bottom[1]);
	}

	template <typename Dtype>
	void LSTMConvUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->shape(1);
		const int x_dim = hidden_dim_ * 4;
		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		const Dtype* flush = bottom[2]->cpu_data();
		Dtype* C = top[0]->mutable_cpu_data();
		Dtype* H = top[1]->mutable_cpu_data();
		//TODO: we can allow different types of activation functions
		for (int n = 0; n < num; n++){
			for (int d = 0; d < hidden_dim_; d++){
				const Dtype i = sigmoid(X[d]);
				const Dtype f = (*flush == 0) ? 0 :
					(*flush * sigmoid(X[1 * hidden_dim_ + d));
				const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
				const Dtype g = tanh(X[3 * hidden_dim_ + d]);
				//here each channel share the same memory 
				//need to be tuned here
				const Dtype c_prev = C_prev[d];
				const Dtype c = f * c_prev + i * g;
				C[d] = c;
				const Dtype tanh_c = tanh(c);
				H[d] = o * tanh_c;
			}
			C += hidden_dim_;
			X += x_dim;
			H += hidden_dim_;
			C_prev += hidden_dim_;
			++flush;
		}
	}

	template <typename Dtype>
	void LSTMConvUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicator.";
		if (!propagate_down[0] && !propagate_down[1]){ continue; }
		
		const int num = bottom[0]->shape(1);
		const int x_dim = hidden_dim_ * 4;
		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		const Dtype* flush = bottom[2]->cpu_data();
		const Dtype* C = top[0]->cpu_data();
		const Dtype* H = top[1]->cpu_data();
		const Dtype* C_diff = top[0]->cpu_diff();
		const Dtype* H_diff = top[1]->cpu_diff();
		Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
		Dtype* X_prev_diff = bottom[1]->mutable_cpu_diff();
		for (int n = 0; n < num; ++n){
			for (int d = 0; d < hidden_dim_; ++d){
				const Dtype i = sigmoid(X[d]);
				const Dtype f = (*flush == 0) ? 0 :
					(*flush * sigmoid(X[1 * hidden_dim_ + d]));
				const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
				const Dtype g = tanh(X[3 * hidden_dim_ + d]);
				const Dtype C_prev = C_prev[d];
			}
		}
	}
}