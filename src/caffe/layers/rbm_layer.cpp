#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe{

	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//TODO: implement CD-k not only CD-1 in current version
		num_iteration_ = this->layer_param_.rbm_param().num_iteration();
		const int num_output = this->layer_param_.rbm_param().num_output();
		//what's the effect of bias_term_ here?
		bias_term_ = this->layer_param_.rbm_param().has_bias_filler();
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.rbm_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		K_ = bottom[0]->count(axis);
		if (this->blobs_.size() > 0){
			LOG(INFO) << "Skipping parameter initialization";
		}
		else{
			//in RBM layer, we have h_bias and v_bias seperately
			if (bias_term_){
				this->blobs_.resize(3);
			}
			else{
				this->blobs_.resize(1);
			}

			//Initialize the weight
			vector<int> weight_shape(2);
			weight_shape[0] = N_;
			weight_shape[1] = K_;
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
			//fill the weights
			shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
				this->layer_param_.rbm_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
			//If necessary, initialize and fill the bias term
			//bias has the same dim with the corresponding data
			if (bias_term_){
				vector<int> h_bias_shape(1, N_);
				vector<int> v_bias_shape(1, K_);
				this->blobs_[1].reset(new Blob<Dtype>(h_bias_shape));
				this->blobs_[2].reset(new Blob<Dtype>(v_bias_shape));
				shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtpye>(
					this->layer_param_.rbm_param().bias_filler());
				bias_filler->Fill(this->blobs_[1].get());
				bias_filler->Fill(this->blobs_[2].get());
			}
		} // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
		//TODO: implement different types of sampling, not just binary 
		//sampling in current version
		sample_type_ = this->layer_param_.rbm_param().sample_type();
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.rbm_param().axis());
		const int new_K = bottom[0]->count(axis);
		CHECK_EQ(K_, new_K)
			<< "Input size incompatible with inner product parameters.";
		//The first "axis" dimensions are independent inner products; the total
		//number of these is M_, the product over these dimensions.
		M_ = bottom[0]->count(0, axis);
		//The top shape will be the bottom shape with the flattened axis dropped,
		// and replaced by a single axis with dimension num_out (N_)
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = N_;
		top[0]->Reshape(top_shape);
		pre_h_.ShareData(*top[0]); 
		pre_v_.ShareData(*bottom[0]);
		cur_h_.Reshape(top_shape);
		positive_state_h_.Reshape(top_shape);
		cur_v_.ReshapeLike(*bottom[0]);
		if (bias_term_){
			vector<int> bias_shape(1, M_);
			bias_multiplier_.Reshape(bias_shape);
			caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
		}
		//output reconstruction loss
		if (top.size() > 1){
			vector<int> loss_shape(0);
			top[2]->Reshape(loss_shape);
		}
	}

	/*
	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_hvh_cpu(){
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		const Dtype* h_bias_data = this->blobs_[1]->cpu_data();
		const Dtype* v_bias_data = this->blobs_[2]->cpu_data();
		const Dtype* pre_h_data = pre_h_.cpu_data();
		Dtype* cur_h_data = cur_h_.mutable_cpu_data();
		Dtype* pre_v_data = pre_v_.cpu_data();
		const int count_h = pre_h_.count();
		const int count_v = pre_v_.count();
		//prop down
		//h: M x N  v: M x K w: N x K
		//dimension of matrix is fixed, so we need to use trans to make
		//the dimension of A and B match
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			pre_h_data, weight_data, (Dtype)0., pre_v_data);
		if (bias_term_){
			//h_bias_multiplier: 1 x N h_bias: 1 x N
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
				v_bias_multiplier_.cpu_data(), v_bias_data, (Dtype)1., pre_v_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_v; i++){
			pre_v_data[i] = sigmoid(pre_v_data[i]);
		}

		//sampling
		caffe_rng_bernoulli<Dtype>(count_v, pre_v_data, pre_v_data);
		//prop up
		//h: M x N  v: M x K w: K x N
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			pre_v_data, weight_data, (Dtype)0., cur_h_data);
		if (bias_term){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				h_bias_multiplier_.cpu_data(), h_bias_data, (Dtype)1., cur_h_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_h; i++){
			cur_h_data[i] = sigmoid(cur_h_data[i]);
		}

		//sampling
		caffe_rng_bernoulli<Dtype>(count_h, cur_h_data, cur_h_data);
	}
	*/

	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_vhvh_cpu(){
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		const Dtype* h_bias_data = this->blobs_[1]->cpu_data();
		const Dtype* v_bias_data = this->blobs_[2]->cpu_data();
		Dtype* pre_h_data = pre_h_.mutable_cpu_data();
		Dtype* cur_h_data = cur_h_.mutable_cpu_data();
		Dtype* positive_state_h_data = positive_state_h_.mutable_cpu_data();
		const Dtype* pre_v_data = pre_v_.cpu_data();
		Dtype* cur_v_data = cur_v_.mutable_cpu_data();
		const int count_h = pre_h_.count();
		const int count_v = cur_v_.count();
		//prop up
		//h: M x N  v: M x K w: N x K
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			pre_v_data, weight_data, (Dtype)0, pre_h_data);
		if (bias_term){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), h_bias_data, (Dtype)1., pre_h_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_h; i++){
			pre_h_data[i] = sigmoid(pred_h_data[i]);
		}
		//sampling
		caffe_rng_bernoulli<Dtype>(count_h, pre_h_data, positive_state_h_data);
		//prop down
		//h: M x N  v: M x K w: N x K
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			positive_state_h_data, weight_data, (Dtype)0., cur_v_data);
		if (bias_term_){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), v_bias_data, (Dtype)1., cur_v_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_v; i++){
			cur_v_data[i] = sigmoid(cur_v_data[i]);
		}
		//sampling 
		//caffe_rng_bernoulli<Dtype>(count_v, cur_v_data, cur_v_data);

		//prop up again
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			cur_v_data, weight_data, (Dtype)0, cur_h_data);
		if (bias_term){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), h_bias_data, (Dtype)1., cur_h_data);
		}

		//sigmoid activation
		for (int i = 0; i < count_h; i++){
			cur_h_data[i] = sigmoid(cur_h_data[i]);
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Gibbs_vhvh_cpu();
		if (top.size() > 1){
			const int count = bottom[0]->count();
			//use cur_v diff for buffer of data
			caffe_sub(count, bottom_data, cur_v_.cpu_data(), cur_v_.mutable_cpu_diff());
			Dtype loss = caffe_cpu_dot(count, cur_v_.cpu_diff(), cur_v_.cpu_diff());
			top[1]->mutable_cpu_data()[0] = loss;
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype* pos_ass_data = pre_h_.mutable_cpu_diff();
		Dtype* neg_ass_data = cur_h_.mutable_cpu_diff();
		const Dtype* pos_v_data = bottom[0]->cpu_data();
		const Dtype* pos_h_data = pre_h_.cpu_data();
		const Dtype* neg_v_data = cur_v_.cpu_data();
		const Dtype* neg_h_data = cur_h_.cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		if (this->param_propagate_down_[0]){
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				pos_h_data, pos_v_data, (Dtype)0., pos_ass_data);
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				neg_h_data, neg_v_data, (Dtype)0., neg_ass_data);
			caffe_sub(N_ * K_, pos_ass_data, neg_ass_data, weight_diff);
		}
		if (propagate_down[0]){
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				top_diff, weight_data, (Dtype)0., bottom_diff);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(RBMLayer)
#endif

INSTANTIATE_CLASS(RBMLayer);
REGISTER_LAYER_CLASS(RBM);
} // namespace caffe
