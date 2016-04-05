
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/added_layers.hpp"

namespace caffe{

template<typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom.size(), 1) << "Locally Conv Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Locally Conv Layer takes a single blob as output.";

	kernel_size_ = this->layer_param_.local_adaptive_dropout_param().kernel_size();
	stride_ = this->layer_param_.local_adaptive_dropout_param().stride();
	pad_ = this->layer_param_.local_adaptive_dropout_param().pad();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	num_output_ = this->layer_param_.local_adaptive_dropout_param().num_output();
	need_act_ = this->layer_param_.local_adaptive_dropout_param().need_act();
	alpha_ = this->layer_param_.local_adaptive_dropout_param().alpha();
	beta_ = this->layer_param_.local_adaptive_dropout_param().beta();
	prob_act_type_ = this->layer_param_.local_adaptive_dropout_param().prob_act_type();
	if (need_act_){
		neuron_act_type_ = this->layer_param_.local_adaptive_dropout_param().neuron_act_type();
	}

	//height of output map
	height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
	width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

	//#inner product parameters
	//no weight sharing
	M_ = num_output_;
	K_ = channels_ * kernel_size_ * kernel_size_;
	N_ = height_out_ * width_out_;

	CHECK_GT(num_output_, 0);
	CHECK_GE(height_, kernel_size_) << "height smaller than kernel size";
	CHECK_GE(width_, kernel_size_) << "width smaller than kernel size";
	//set bias term
	bias_term_ = this->layer_param_.local_adaptive_dropout_param().bias_term();

	//Check if we need to set up weights
	if (this->blobs_.size() > 0){
		LOG(INFO) << "Skipping parameter initialization";
	}
	else{
		if (bias_term_){
			this->blobs_.resize(2);
		}
		else{
			this->blobs_.resize(1);
		}
	}

	//Initialize the weight
	this->blobs_[0].reset(new Blob<Dtype>(num_output_, 1, K_, N_));
	//fill the weights
	shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
		this->layer_param_.local_adaptive_dropout_param().weight_filler()));
	weight_filler->Fill(this->blobs_[0].get());
	//if necessary, initialize and fill the bias term
	if (bias_term_){
		this->blobs_[1].reset(new Blob<Dtype>(1, 1, M_, N_));
		shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(
			this->layer_param_.local_adaptive_dropout_param().bias_filler()));
		bias_filler->Fill(this->blobs_[1].get());
	}
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
		<< " weights.";
	//TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id){
		CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
		CHECK_EQ(channels_, bottom[bottom_id]->channels()) << "Inputs must have same channels.";
		CHECK_EQ(height_, bottom[bottom_id]->height()) << "Inputs must have same height.";
		CHECK_EQ(width_, bottom[bottom_id]->width()) << "Inputs must have same width.";
	}

	//shape the tops
	for (int top_id = 0; top_id < top.size(); ++top_id){
		top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
	}

	//The im2col result buffer only image at a time to avoid 
	//overly large memory usage.
	col_buffer_.Reshape(1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);
	//initialize cache for prob weight
	prob_weight_.Reshape(this->blobs_[0]->shape());
	//initialize cache for probability
	prob_vec_.Reshape(num_, num_output_, height_out_, width_out_);
	//initialize cache for mask
	mask_vec_.Reshape(prob_vec_.shape());
	//initialize cache for unacted hidden units
	unact_hidden_.Reshape(prob_vec_.shape());
}


template <typename Dtype>
//inline: only self and friend class can call
inline Dtype sigmoid(Dtype x){
	return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype relu(Dtype x){
	return std::max(x, Dtype(0));
}

template <typename Dtype>
inline void activate(const int n,const  Dtype* in, Dtype* out,
	LocalAdaptiveDropoutParameter_ActType act_type){
	switch (act_type){
	case caffe::LocalAdaptiveDropoutParameter_ActType_SIGMOID:
		for (int i = 0; i < n; i++){
			out[i] = sigmoid<Dtype>(in[i]);
		}
		break;
	case caffe::LocalAdaptiveDropoutParameter_ActType_RELU:
		for (int i = 0; i < n; i++){
			out[i] = relu<Dtype>(in[i]);
		}
		break;
	default:
		LOG(FATAL) << "Unkown activate function.";
	}
}

template <typename Dtype>
inline void common_mul(const int n, const Dtype* a, const unsigned int* b, Dtype* y){
	for (int i = 0; i < n; i++){
		y[i] = a[i] * Dtype(b[i]);
	}
}

template <typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Dtype* x_data = col_buffer_.mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* unact_data = unact_hidden_.mutable_cpu_data();

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);

	Dtype* prob_weight = this->prob_weight_.mutable_cpu_data();
	for (int i = 0; i < this->blobs_[0]->count(); i++){
		prob_weight[i] = alpha_ * weight[i] + beta_;
	}

	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, K_, N_);
	Dtype* prob = this->prob_vec_.mutable_cpu_data();
	//get probabilities for all output units
	for (int n = 0; n < num_; n++){
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		for (int m = 0; m < num_output_; m++){
			caffe_mul(K_*N_, x_data, prob_weight + this->prob_weight_.offset(m),
				intermediate.mutable_cpu_data());

			//like add up?
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_, (Dtype)1.,
				E.cpu_data(), intermediate.cpu_data(),
				(Dtype)0., prob + prob_vec_.offset(n,m));
		}
	}
	//activate probability
	activate<Dtype>(prob_vec_.count(), prob, prob, prob_act_type_);

	//get unacted hidden values
	intermediate.Reshape(1, 1, K_, N_);
	for (int n = 0; n < num_; n++){
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);
		for (int m = 0; m < num_output_; m++){
			caffe_mul(K_*N_, x_data, weight + this->blobs_[0]->offset(m),
				intermediate.mutable_cpu_data());

			//like add up?
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_, (Dtype)1.,
				E.cpu_data(), intermediate.cpu_data(),
				(Dtype)0., unact_data + unact_hidden_.offset(n, m));
		}
		if (bias_term_){
			caffe_add(M_ * N_, this->blobs_[1]->cpu_data(),
				 unact_data + unact_hidden_.offset(n),
				 unact_data + unact_hidden_.offset(n));
		}
	}
	//if need activation, activate top data
	if (need_act_){
		activate<Dtype>(top[0]->count(), unact_hidden_.cpu_data(), top[0]->mutable_cpu_data(),
			neuron_act_type_);
	}
	else{
		caffe_copy<Dtype>(top[0]->count(), unact_hidden_.cpu_data(), top[0]->mutable_cpu_data());
	}
	//when training, sample from probability to generate mask
	//otherwise, just multiply the expectation(probability)
	if (this->phase_ == TRAIN){
		DCHECK(prob_vec_.count() == mask_vec_.count());
		DCHECK(prob_vec_.count() == top[0]->count());
		unsigned int* rand_vec = this->mask_vec_.mutable_cpu_data();
		caffe_rng_bernoulli<Dtype>(prob_vec_.count(), prob_vec_.cpu_data(), rand_vec);
		common_mul<Dtype>(mask_vec_.count(), top_data, mask_vec_.cpu_data(), top_data);
	}
	else{
		caffe_mul<Dtype>(prob_vec_.count(), top_data, prob_vec_.cpu_data(), top_data);
	}
}

template<typename Dtype>
inline void SigmoidBackward(const int n, const Dtype* in_diff,
	const Dtype* unact_data, Dtype* out_diff){
	for(int i= 0; i<n ; i++){
		const Dtype sigmoid_x = 1. / (1. + exp(- unact_data[i]));
		out_diff[i] = in_diff[i] * sigmoid_x * (1 - sigmoid_x);
	}
}

template <typename Dtype>
inline void ReLUBackward(const int n, const Dtype* in_diff,
	const Dtype* in_data, Dtype* out_diff){
	for (int i = 0; i<n;i++){
		out_diff[i] = in_diff[i] * (in_data[i] > 0);
	}
}

template <typename Dtype>
inline void ActBackward(const int n, const Dtype* in_diff,
	const Dtype* in_data, Dtype* out_diff, LocalAdaptiveDropoutParameter_ActType act_type){
	switch (act_type)
	{
	case caffe::LocalAdaptiveDropoutParameter_ActType_RELU:
		ReLUBackward<Dtype>(n, in_diff, in_data, out_diff);
		break;
	case caffe::LocalAdaptiveDropoutParameter_ActType_SIGMOID:
		SigmoidBackward<Dtype>(n, in_diff, in_data, out_diff);
		break;
	default:
		LOG(FATAL) << "unknown act function type.";
		break;
	}
} 

template <typename Dtype>
void LocalAdaptiveDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,

	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* x_data = col_buffer_.mutable_cpu_data();
	Dtype* x_diff = col_buffer_.mutable_cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = NULL;

	//backward through dropout
	const unsigned int* mask_vec = this->mask_vec_.cpu_data();
	common_mul<Dtype>(top[0]->count(), top_diff, mask_vec, prob_vec_.mutable_cpu_diff());
	//backward through non-linear activation
	const Dtype* in_data = unact_hidden_.cpu_data();
	ActBackward<Dtype>(top[0]->count(), prob_vec_.cpu_diff(), in_data,
		unact_hidden_.mutable_cpu_diff(), neuron_act_type_);

	//backward throught locally-connected part
	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, 1, N_);

	Blob<Dtype> xt;
	const Dtype* unact_diff = unact_hidden_.cpu_diff();
	//weight buffer
	xt.Reshape(1, 1, K_, N_);
	Dtype* xt_data = xt.mutable_cpu_data();
	//gradient wrt bias
	if (bias_term_){
		//Output = W * X + b
		//db = do
		bias_diff = this->blobs_[1]->mutable_cpu_diff();
		memset(bias_diff, 0, sizeof(Dtype)* this->blobs_[1]->count());
		for (int n = 0; n < num_; ++n){
			caffe_add(M_* N_, bias_diff,
				unact_diff + unact_hidden_.offset(n),
				bias_diff);
		}
	}

	memset(weight_diff, 0, sizeof(Dtype)* this->blobs_[0]->count());
	for (int n = 0; n < num_; n++){
		//image 2 column
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_,
			x_data);


		//gradient wrt weight
		for (int m = 0; m < num_output_; m++){
			Dtype* filter_weight_diff = weight_diff + this->blobs_[0]->offset(m);
			for (int k = 0; k < K_; k++){
				caffe_mul(N_,  unact_diff + unact_hidden_.offset(n, m),
					x_data + col_buffer_.offset(0, k), xt_data + xt.offset(0, 0, k));
			}
			//cumulate through different channels and nums
			caffe_cpu_axpby(K_ * N_, Dtype(1.0), xt_data, Dtype(1.0), filter_weight_diff);
		}
		//gradient wrt bottom data
		if (propagate_down[0]){
			memset(x_diff, 0, col_buffer_.count() * sizeof(Dtype));
			for (int m = 0; m < num_output_; m++){
				for (int k = 0; k < K_; k++){
					caffe_mul(N_, unact_diff + unact_hidden_.offset(n, m),
						weight + this->blobs_[0]->offset(m, 0, k),
						intermediate.mutable_cpu_data());
					caffe_cpu_axpby(N_, (Dtype)1.,
						intermediate.cpu_data(), Dtype(1.0),
						x_diff + col_buffer_.offset(0, k));
				}
			}
			//col2im back to data
			col2im_cpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
				pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));
		}
	}
}
#ifdef CPU_ONLY
STUB_GPU(LocalAdaptiveDropoutLayer);
#endif

INSTANTIATE_CLASS(LocalAdaptiveDropoutLayer);
REGISTER_LAYER_CLASS(LocalAdaptiveDropout);
}
