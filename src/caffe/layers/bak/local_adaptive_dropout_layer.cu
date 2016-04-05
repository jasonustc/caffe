
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/added_layers.hpp"

namespace caffe{
template <typename Dtype>
__global__ void local_update1_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
	Dtype* data_R, const int filter_num, const int location_num, const int output_num){
	int total = filter_num * location_num * output_num;
	CUDA_KERNEL_LOOP(index, total){
		int p = index % location_num;
		int n = (index / location_num) % filter_num;
		int q = (index / location_num) / filter_num;
		data_R[index] += data_A[q*location_num + p] * data_B[n*location_num + p];
	}
}

template <typename Dtype>
void local_update1_gpu(const Dtype* data_A, const Dtype* data_B,
	Dtype* data_R, const int filter_num,
	const int location_num, const int output_num){
	//data_A is output_num x location_num
	//data_B is filter_num x location_num
	//data_R is output_num x filter_num x location_num, the update performed is Rqnp += Aap * Bnp
	local_update1_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(filter_num * location_num * output_num),
		CAFFE_CUDA_NUM_THREADS >> >(data_A, data_B, data_R, filter_num, location_num, output_num);
	CUDA_POST_KERNEL_CHECK;
}

//explicit instantiation
template void local_update1_gpu<float>(const float* data_A, const float* data_B,
	float* data_R, const int filter_num,
	const int location_num, const int output_num);

template void local_update1_gpu<double>(const double* data_A, const double* data_B,
	double* data_R, const int filter_num,
	const int location_num, const int output_num);

template <typename Dtype>
__global__ void local_update2_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
	Dtype* data_R, const int filter_num, const int location_num, const int output_num){
	int total = filter_num * location_num;
	CUDA_KERNEL_LOOP(index, total){
		int p = index % location_num;
		int n = index / location_num;
		for (int q = 0; q < output_num; q++){
			data_R[index] += data_A[q*location_num + p] * data_B[(q*filter_num + n) * location_num + p];
		}
	}
}

template <typename Dtype>
void local_update2_gpu(const Dtype* data_A, const Dtype* data_B,
	Dtype* data_R, const int filter_num, const int location_num, const int output_num){
	//data_A is output_num x location_num
	//data_B is output_num x filter_num x location_num
	//data_R is filter_num x location_num, the update performed is Rnp += \sum_q(Aqp * Bqnp)

	//NOLINT_NEXT_LINE(whitespace/operator)
	local_update2_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(filter_num * location_num),
		CAFFE_CUDA_NUM_THREADS >> >(data_A, data_B, data_R, filter_num, location_num, output_num);
	CUDA_POST_KERNEL_CHECK;
}

//explicit instantiation
template void local_update2_gpu<float>(const float* data_A, const float* data_B,
	float* data_R, const int filter_num, const int location_num, const int output_num);

template void local_update2_gpu<double>(const double* data_A, const double* data_B,
	double* data_R, const int filter_num, const int location_num, const int output_num);

template <typename Dtype>
__global__ void SigmoidActivate(const int n, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = 1. / (1. + exp(-in[index]));
	}
}

template <typename Dtype>
__global__ void ReluActivate(const int n, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = in[index]> 0 ? in[index] : 0;
	}
}

template <typename Dtype>
inline void activate(const int n, const Dtype* in, Dtype* out,
	LocalAdaptiveDropoutParameter_ActType act_type){
	switch (act_type){
	case caffe::LocalAdaptiveDropoutParameter_ActType_SIGMOID:
		SigmoidActivate<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
		break;
	case caffe::LocalAdaptiveDropoutParameter_ActType_RELU:
		ReluActivate<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
		break;
	default:
		LOG(FATAL) << "Unkown activate function.";
	}
}


template <typename Dtype>
__global__ void caffe_mult_and_add_scalar(const int n, const Dtype* in, Dtype* out,
	const Dtype alpha, const Dtype beta){
	CUDA_KERNEL_LOOP(index, n){
		out[index] = alpha * in[index] + beta;
	}
}

///@ brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Dtype* x_data = col_buffer_.mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* unact_data = unact_hidden_.mutable_gpu_data();

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	//value in constant filler
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);

	//get probability weight
	Dtype* prob_weight = this->prob_weight_.mutable_gpu_data();
	caffe_mult_and_add_scalar<Dtype> << <CAFFE_GET_BLOCKS(prob_weight_.count()),
		CAFFE_CUDA_NUM_THREADS >> >(prob_weight_.count(), weight, prob_weight,
		alpha_, beta_);
	//compute probability
	Blob<Dtype> intermediate;
	Dtype* prob_data = this->prob_vec_.mutable_gpu_data();
	intermediate.Reshape(1, 1, K_, N_);
	for (int n = 0; n < num_; n++){
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		for (int m = 0; m < num_output_; m++){
			caffe_gpu_mul(K_ * N_, x_data, prob_weight + prob_weight_.offset(m),
				intermediate.mutable_gpu_data());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
				(Dtype)1., E.gpu_data(), intermediate.gpu_data(),
				(Dtype)0., prob_data + prob_vec_.offset(n, m));
		}
		if (bias_term_){
			caffe_gpu_add(M_ * N_, this->blobs_[1]->gpu_data(),
				prob_data + prob_vec_.offset(n),
				prob_data + prob_vec_.offset(n));
		}
	}
	//activate probability
	activate<Dtype>(prob_vec_.count(), prob_vec_.gpu_data(), prob_vec_.mutable_gpu_data(), prob_act_type_);

	//locally inner-product for hidden units
	//store in unact_hidden_
	intermediate.Reshape(1, 1, K_, N_);
	for (int n = 0; n < num_; n++){
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		for (int m = 0; m < num_output_; m++){
			caffe_gpu_mul(K_ * N_, x_data, weight + this->blobs_[0]->offset(m),
				intermediate.mutable_gpu_data());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
				(Dtype)1., E.gpu_data(), intermediate.gpu_data(),
				(Dtype)0., unact_data + unact_hidden_.offset(n, m));
		}
		if (bias_term_){
			caffe_gpu_add(M_ * N_, this->blobs_[1]->gpu_data(),
				unact_data + unact_hidden_.offset(n),
				unact_data + unact_hidden_.offset(n));
		}
	}
	//activate hidden units
	activate<Dtype>(top[0]->count(), unact_hidden_.gpu_data(), top_data, neuron_act_type_);
	CUDA_POST_KERNEL_CHECK;
	if (this->phase_ == TRAIN){
		caffe_gpu_rng_bernoulli<Dtype>(prob_vec_.count(), prob_vec_.gpu_data(), mask_vec_.mutable_gpu_data());
		caffe_gpu_mul_b<Dtype>(prob_vec_.count(), top[0]->gpu_data(), mask_vec_.gpu_data(), 
			top[0]->mutable_gpu_data());
	}
	else{
		caffe_gpu_mul<Dtype>(prob_vec_.count(), top[0]->gpu_data(), prob_vec_.gpu_data(), 
			top[0]->mutable_gpu_data());
	}
}

template<typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
	const Dtype* unact_data, Dtype* out_diff){
	CUDA_KERNEL_LOOP(index, n){
		const Dtype sigmoid_x = 1. / (1. + exp(-unact_data[index]));
		out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
	}
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
	const Dtype* in_data, Dtype* out_diff){
	CUDA_KERNEL_LOOP(index, n){
		out_diff[index] = in_diff[index] * (in_data[index] > 0);
	}
}

template <typename Dtype>
inline void ActBackward(const int n, const Dtype* in_diff,
	const Dtype* in_data, Dtype* out_diff, LocalAdaptiveDropoutParameter_ActType act_type){
	switch (act_type)
	{
	case caffe::LocalAdaptiveDropoutParameter_ActType_RELU:
		ReLUBackward<Dtype ><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, in_diff, in_data, out_diff);
		break;
	case caffe::LocalAdaptiveDropoutParameter_ActType_SIGMOID:
		SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, in_diff, in_data, out_diff);
		break;
	default:
		LOG(FATAL) << "unknown act function type.";
		break;
	}
} 

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
	const unsigned int* mask, const float scale, Dtype* out_diff){
	CUDA_KERNEL_LOOP(index, n){
		out_diff[index] = in_diff[index] * scale * mask[index];
	}
}

///@brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* x_data = col_buffer_.mutable_gpu_data();
	Dtype* x_diff = col_buffer_.mutable_gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	Dtype* bias_diff = NULL;

	//backward through dropout
	//store in prob_diff
	const unsigned int* rand_vec = this->mask_vec_.mutable_gpu_data();
	DropoutBackward<Dtype> << <CAFFE_GET_BLOCKS(mask_vec_.count()), CAFFE_CUDA_NUM_THREADS >> >
		(mask_vec_.count(), top_diff, rand_vec, (Dtype)1., prob_vec_.mutable_gpu_diff());
	//backward through non-linear activation
	const Dtype* in_data = unact_hidden_.gpu_data();
	ActBackward(top[0]->count(), prob_vec_.gpu_diff(), in_data, unact_hidden_.mutable_gpu_diff(),
		neuron_act_type_);

	//backward through locally-connect
	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, 1, N_);

	//diff of bias
	Blob<Dtype> xt;
	xt.Reshape(1, 1, K_, N_);
	Dtype* xt_data = xt.mutable_gpu_data();
	const Dtype* unact_diff = unact_hidden_.gpu_diff();
	if (bias_term_){
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
		CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype)* this->blobs_[1]->count()));
		for (int n = 0; n < num_; ++n){
			caffe_gpu_add(M_ * N_, bias_diff,
				unact_diff + unact_hidden_.offset(n),
				bias_diff);
		}
	}

	Blob<Dtype> buf;
	buf.Reshape(1, 1, K_, N_);
	Dtype* buf_data = buf.mutable_gpu_data();
	CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype)* this->blobs_[0]->count()));
	for (int n = 0; n < num_; n++){
		//diff of weight
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		local_update1_gpu(unact_diff + unact_hidden_.offset(n), x_data, weight_diff, K_, N_, M_);
		//diff of bottom
		if (propagate_down[0]){
			CUDA_CHECK(cudaMemset(x_diff, 0, col_buffer_.count() * sizeof(Dtype)));
			local_update2_gpu(unact_diff + unact_hidden_.offset(n), weight, x_diff, K_, N_, M_);

			//col2im back to the data
			col2im_gpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
				pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalAdaptiveDropoutLayer);

}//namespace caffe
