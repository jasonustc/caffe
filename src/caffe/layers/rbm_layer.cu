#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe{

	template <typename Dtype>
	__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out){
		CUDA_KERNEL_LOOP(index, n){
			out[index] = 1. / (1. + exp(-in[index]));
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_vhvh_gpu(){
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		const Dtype* h_bias_data = this->blobs_[1]->gpu_data();
		const Dtype* v_bias_data = this->blobs_[2]->gpu_data();
		Dtype* pre_h_data = pre_h_.mutable_gpu_data();
		Dtype* cur_h_data = cur_h_.mutable_gpu_data();
		Dtype* positive_state_h_data = positive_state_h_.mutable_gpu_data();
		const Dtype* pre_v_data = pre_v_.gpu_data();
		Dtype* cur_v_data = cur_v_.mutable_gpu_data();
		const int count_h = pre_h_.count();
		const int count_v = cur_v_.count();
		//prop up
		//h: M x N  v: M x K w: N x K
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			pre_v_data, weight_data, (Dtype)0, pre_h_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), h_bias_data, (Dtype)1., pre_h_data);
		}
		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_h), CAFFE_CUDA_NUM_THREADS >> >(
			count_h, pre_h_data, pre_h_data);
		//sampling
		caffe_gpu_rng_bernoulli<Dtype>(count_h, pre_h_data, positive_state_h_data);
		//prop down
		//h: M x N  v: M x K w: N x K
		//TODO: need to convert the data type of state_h to Dtype
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			positive_state_h_data, weight_data, (Dtype)0., cur_v_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), v_bias_data, (Dtype)1., cur_v_data);
		}
		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_v), CAFFE_CUDA_NUM_THREADS >> >(
			count_v, cur_v_data, cur_v_data);
		//sampling 
		//caffe_rng_bernoulli<Dtype>(count_v, cur_v_data, cur_v_data);

		//prop up again
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			cur_v_data, weight_data, (Dtype)0, cur_h_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), h_bias_data, (Dtype)1., cur_h_data);
		}

		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_h), CAFFE_CUDA_NUM_THREADS >> >(
			count_h, cur_h_data, cur_h_data);
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//top[0] shares data with pre_h_ data
		Gibbs_vhvh_gpu();
		//output loss
		if (top.size() > 1){
			const int count = bottom[0]->count();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			const Dtype* cur_v_data = cur_v_.gpu_data();
			Dtype* tmp_data = cur_v_.mutable_gpu_diff();
			caffe_gpu_sub<Dtype>(count, bottom_data, cur_v_data, tmp_data);
			Dtype loss;
			caffe_gpu_dot<Dtype>(count, tmp_data, tmp_data, &loss);
			top[1]->mutable_cpu_data()[0] = loss;
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype* pos_ass_data = pre_h_.mutable_gpu_diff();
		Dtype* neg_ass_data = cur_h_.mutable_gpu_diff();
		const Dtype* pos_v_data = bottom[0]->gpu_data();
		const Dtype* pos_h_data = pre_h_.gpu_data();
		const Dtype* neg_v_data = cur_v_.gpu_data();
		const Dtype* neg_h_data = cur_h_.gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		if (propagate_down[0]){
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				pos_h_data, pos_v_data, (Dtype)0., pos_ass_data);
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				neg_h_data, neg_v_data, (Dtype)0., neg_ass_data);
			caffe_gpu_sub(N_ * K_, pos_ass_data, neg_ass_data, weight_diff);
		}
		if (propagate_down[0]){
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				top_diff, weight_data, (Dtype)0., bottom_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(RBMLayer);
}
