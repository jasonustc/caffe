/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/10/4
** desc: DecodingRecurrentLayer(CPU)
*********************************************************************************/
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
string DRecurrentLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void DRecurrentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_GE(bottom[0]->num_axes(), 2)
		<< "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
	//timesteps
	LOG(INFO) << "bottom[0] shape: " << bottom[0]->shape_string();
	T_ = bottom[0]->shape(0);
	//streams
	N_ = bottom[0]->shape(1);
	//sequence length
	len_seq_ = this->layer_param_.recurrent_param().sequence_length();
	CHECK_GT(len_seq_, 0)<< "sequence length must be positive";
	CHECK(T_ % len_seq_ == 0) << "T_ must be divided by sequence length";
	LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
		<< T_ << " timesteps of " << N_ << " independent streams.";

	//the indicator
	int num_output = this->layer_param_.recurrent_param().num_output();
	LOG(INFO) << "bottom[1] shape: " << bottom[1]->shape_string();
	CHECK_EQ(T_ / len_seq_, bottom[1]->shape(0));
	CHECK_EQ(N_, bottom[1]->shape(1));
	CHECK_EQ(num_output, bottom[1]->shape(2));
	
	CHECK(bottom[1]->shape() == bottom[2]->shape()) << "h and c should have the same shape";

	// If provided, bottom[3] is a static input to the recurrent net.
	static_input_ = (bottom.size() > 3);
	if (static_input_) {
		CHECK_GE(bottom[3]->num_axes(), 1);
		CHECK_EQ(N_, bottom[3]->shape(0));
	}

	// Create a NetParameter; setup the inputs that aren't unique to particular
	// recurrent architectures.
	NetParameter net_param;
	net_param.set_force_backward(true);

	//connect the input this recurrent layer to the 
	//input of LSTM Net
	net_param.add_input("x");
	BlobShape input_shape;
	for (int i = 0; i < bottom[0]->num_axes(); ++i) {
		input_shape.add_dim(bottom[0]->shape(i));
	}
//	//repeated message can be added 
	net_param.add_input_shape()->CopyFrom(input_shape);

	input_shape.Clear();
	for (int i = 0; i < bottom[1]->num_axes(); ++i) {
		input_shape.add_dim(bottom[1]->shape(i));
	}
	//input is the required domain of a net?
	//h: num_sequences x N x hidden_dim
	net_param.add_input("h");
	net_param.add_input_shape()->CopyFrom(input_shape);

	//c: num_sequences x N x hidden_dim
	net_param.add_input("c");
	net_param.add_input_shape()->CopyFrom(input_shape);

	if (static_input_) {
		input_shape.Clear();
		for (int i = 0; i < bottom[3]->num_axes(); ++i) {
			input_shape.add_dim(bottom[3]->shape(i));
		}
		net_param.add_input("x_static");
		net_param.add_input_shape()->CopyFrom(input_shape);
	}

	// Call the child's FillUnrolledNet implementation to specify the unrolled
	// recurrent architecture.
	this->FillUnrolledNet(&net_param);

	// Prepend this layer's name to the names of each layer in the unrolled net.
	const string& layer_name = this->layer_param_.name();
	if (layer_name.size() > 0) {
		for (int i = 0; i < net_param.layer_size(); ++i) {
			LayerParameter* layer = net_param.mutable_layer(i);
			layer->set_name(layer_name + "_" + layer->name());
		}
	}

	// Create the unrolled net.
	//call the Init() function to build the net.
	unrolled_net_.reset(new Net<Dtype>(net_param));
	//if need to output debug info
	unrolled_net_->set_debug_info(
		this->layer_param_.recurrent_param().debug_info());

	// Setup pointers to the inputs.
	x_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("x").get());
	h_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("h").get());
	c_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("c").get());
	if (static_input_) {
		x_static_input_blob_ =
			CHECK_NOTNULL(unrolled_net_->blob_by_name("x_static").get());
	}

	// Setup pointers to outputs.
	vector<string> output_names;
	OutputBlobNames(&output_names);
	//change here to make sure we can output c_T and h_T indepentently
	CHECK_EQ(top.size(), output_names.size())
		<< "OutputBlobNames must provide an output blob name for each top.";
	output_blobs_.resize(output_names.size());
	for (int i = 0; i < output_names.size(); ++i) {
		//blob_by_name: get blob by name
		output_blobs_[i] =
			CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
	}

	// We should have 2 inputs (x and cont), plus a number of recurrent inputs,
	// plus maybe a static input.
	//x, c, h, static_input_
	CHECK_EQ(1 + 2 + static_input_,
		unrolled_net_->input_blobs().size());

	// This layer's parameters are any parameters in the layers of the unrolled
	// net. We only want one copy of each parameter, so check that the parameter
	// is "owned" by the layer, rather than shared with another.
	// Here by set name in param should be able to share parameters with other LSTM
	// because it is corresponding with index in blobs
	this->blobs_.clear();
	for (int i = 0; i < unrolled_net_->params().size(); ++i) {
		if (unrolled_net_->param_owners()[i] == -1) {
			LOG(INFO) << "Adding parameter " << i << ": "
				<< unrolled_net_->param_display_names()[i];
			this->blobs_.push_back(unrolled_net_->params()[i]);
		}
	}
	// Check that param_propagate_down is set for all of the parameters in the
	// unrolled net; set param_propagate_down to true in this layer.
	for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
		for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
			CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
				<< "param_propagate_down not set for layer " << i << ", param " << j;
		}
	}
	this->param_propagate_down_.clear();
	this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void DRecurrentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top.size(), output_blobs_.size());
	for (int i = 0; i < top.size(); ++i) {
		top[i]->ReshapeLike(*output_blobs_[i]);
		top[i]->ShareData(*output_blobs_[i]);
		top[i]->ShareDiff(*output_blobs_[i]);
	}

	//connect input of recurrent net with bottom blobs
	x_input_blob_->ReshapeLike(*bottom[0]);
	x_input_blob_->ShareData(*bottom[0]);
	x_input_blob_->ShareDiff(*bottom[0]);

	h_input_blob_->ReshapeLike(*bottom[1]);
	h_input_blob_->ShareData(*bottom[1]);
	h_input_blob_->ShareDiff(*bottom[1]);

	c_input_blob_->ReshapeLike(*bottom[2]);
	c_input_blob_->ShareData(*bottom[2]);
	c_input_blob_->ShareDiff(*bottom[2]);

	if (static_input_) {
		x_static_input_blob_->ShareData(*bottom[3]);
		x_static_input_blob_->ShareDiff(*bottom[3]);
	}
}

template <typename Dtype>
void DRecurrentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeightData();
  }

  unrolled_net_->ForwardPrefilled();
}

template <typename Dtype>
void DRecurrentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  unrolled_net_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DRecurrentLayer, Forward);
#endif

INSTANTIATE_CLASS(DRecurrentLayer);

}  // namespace caffe
