
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
__global__ void DropWeight(const int n, const Dtype* in,
	const unsigned int* mask, const unsigned int threshold, const float scale,
	Dtype* out){
	//what's the usage of index here?
	CUDA_KERNEL_LOOP(index, n){
		out[index] = in[index] * (mask[index] > threshold) * scale;
	}
}

///@ brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalDropConnectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Dtype* x_data = col_buffer_.mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	//value in constant filler
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);
	//dropout weight
	Dtype* dropped_weight = this->dropped_weight_.mutable_gpu_data();
	if (this->phase_ == TRAIN){
		unsigned int* weight_mask = this->weight_mask_.mutable_gpu_data();
		caffe_gpu_rng_uniform(this->blobs_[0]->count(), weight_mask);
		DropWeight<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			this->blobs_[0]->count(), weight, weight_mask, uint_thres_, scale_,
			dropped_weight);
	}
	else{
		caffe_copy<Dtype>(this->blobs_[0]->count(), weight, dropped_weight);
	}

	//locally inner-product for hidden units
	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, K_, N_);
	for (int n = 0; n < num_; n++){
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		for (int m = 0; m < num_output_; m++){
			caffe_gpu_mul(K_ * N_, x_data, dropped_weight + this->dropped_weight_.offset(m),
				intermediate.mutable_gpu_data());
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
				(Dtype)1., E.gpu_data(), intermediate.gpu_data(),
				(Dtype)0., top_data + top[0]->offset(n, m));
		}
		if (bias_term_){
			caffe_gpu_add(M_ * N_, this->blobs_[1]->gpu_data(),
				top_data + top[0]->offset(n),
				top_data + top[0]->offset(n));
		}
	}
}

///@brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalDropConnectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* x_data = col_buffer_.mutable_gpu_data();
	Dtype* x_diff = col_buffer_.mutable_gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	Dtype* bias_diff = NULL;

	//backward through locally-connect
	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, 1, N_);

	//diff of bias
	Blob<Dtype> xt;
	xt.Reshape(1, 1, K_, N_);
	Dtype* xt_data = xt.mutable_gpu_data();
	if (bias_term_){
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
		CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype)* this->blobs_[1]->count()));
		for (int n = 0; n < num_; ++n){
			caffe_gpu_add(M_ * N_, bias_diff,
				top_diff + top[0]->offset(n),
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
		//cumulate of each samples
		local_update1_gpu(top_diff + top[0]->offset(n), x_data, weight_diff, K_, N_, M_);
		DropWeight<Dtype> << <CAFFE_GET_BLOCKS(weight_mask_.count()), CAFFE_CUDA_NUM_THREADS >> >(
			weight_mask_.count(), weight_diff, weight_mask_.gpu_data(), uint_thres_, scale_,
			this->blobs_[0]->mutable_gpu_data());

		//diff of bottom
		if (propagate_down[0]){
			CUDA_CHECK(cudaMemset(x_diff, 0, col_buffer_.count() * sizeof(Dtype)));
			local_update2_gpu(top_diff + top[0]->offset(n), weight, x_diff, K_, N_, M_);

			//col2im back to the data
			col2im_gpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
				pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalDropConnectLayer);

}//namespace caffe
