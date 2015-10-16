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
string RecurrentLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void RecurrentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//if this is an encoding-decoding structure or not
	const bool decode = this->layer_param_.recurrent_param().decode();
	CHECK_GE(bottom[0]->num_axes(), 2)
		<< "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
	//timesteps
	LOG(INFO) << "bottom[0] shape: " << bottom[0]->shape_string();
	T_ = bottom[0]->shape(0);
	//streams
	N_ = bottom[0]->shape(1);
	LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
		<< T_ << " timesteps of " << N_ << " independent streams.";

	//the cont indicator
	LOG(INFO) << "bottom[1] shape: " << bottom[1]->shape_string();
	CHECK_EQ(bottom[1]->num_axes(), 2)
		<< "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
	CHECK_EQ(T_, bottom[1]->shape(0));
	CHECK_EQ(N_, bottom[1]->shape(1));

	// If provided, bottom[2] is a static input to the recurrent net.
	static_input_ = (bottom.size() > 2);
	if (static_input_) {
		CHECK_GE(bottom[2]->num_axes(), 1);
		CHECK_EQ(N_, bottom[2]->shape(0));
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
	//repeated message can be added 
	net_param.add_input_shape()->CopyFrom(input_shape);

	input_shape.Clear();
	input_shape.add_dim(1);
	//#streams
	for (int i = 0; i < bottom[1]->num_axes(); ++i) {
		input_shape.add_dim(bottom[1]->shape(i));
	}
	//cont: 1 x T x N
	net_param.add_input("cont");
	net_param.add_input_shape()->CopyFrom(input_shape);

	if (static_input_) {
		input_shape.Clear();
		for (int i = 0; i < bottom[2]->num_axes(); ++i) {
			input_shape.add_dim(bottom[2]->shape(i));
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
	cont_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("cont").get());
	if (static_input_) {
		x_static_input_blob_ =
			CHECK_NOTNULL(unrolled_net_->blob_by_name("x_static").get());
	}

	// Setup pointers to paired recurrent inputs/outputs.
	vector<string> recur_input_names;
	RecurrentInputBlobNames(&recur_input_names);
	vector<string> recur_output_names;
	RecurrentOutputBlobNames(&recur_output_names);
	vector<string> recur_end_names;
	const int num_recur_blobs = recur_input_names.size();
	//number of recurrent input blobs must be equal to number of output blobs
	CHECK_EQ(num_recur_blobs, recur_output_names.size());
	recur_input_blobs_.resize(num_recur_blobs);
	recur_output_blobs_.resize(num_recur_blobs);
	if (decode){
		//h_Ts
		EndSequenceBlobNames(&recur_end_names);
		CHECK_EQ(num_recur_blobs, recur_end_names.size());
		all_end_seq_h_ = CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_end_names[0]).get());
		//c_Ts
		all_end_seq_c_ = CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_end_names[1]).get());
	}
	for (int i = 0; i < recur_input_names.size(); ++i) {
		//make sure they are defined and built in recurr architec
		recur_input_blobs_[i] =
			CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_input_names[i]).get());
		recur_output_blobs_[i] =
			CHECK_NOTNULL(unrolled_net_->blob_by_name(recur_output_names[i]).get());
	}

	// Setup pointers to outputs.
	vector<string> output_names;
	OutputBlobNames(&output_names);
	//change here to make sure we can output c_T and h_T indepentently
	CHECK_GE(top.size(), output_names.size())
		<< "OutputBlobNames must provide an output blob name for each top.";
	output_blobs_.resize(output_names.size());
	for (int i = 0; i < output_names.size(); ++i) {
		//blob_by_name: get blob by name
		output_blobs_[i] =
			CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
	}

	// We should have 2 inputs (x and cont), plus a number of recurrent inputs,
	// plus maybe a static input.
	CHECK_EQ(2 + num_recur_blobs + static_input_,
		unrolled_net_->input_blobs().size());

	// This layer's parameters are any parameters in the layers of the unrolled
	// net. We only want one copy of each parameter, so check that the parameter
	// is "owned" by the layer, rather than shared with another.
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

	// here since they are not bottom of any other layer, so they are top layers
	// so loss weight is stored in the cpu_diff
	// loss weight is set to 0 means we don't backpropagate diff of this layer down
	// Set the diffs of recurrent outputs to 0 -- we can't backpropagate across
	// batches.
	//if this is a decode LSTM, the error of c_T and h_T must be propagated down
	//because it is correlated with decoder LSTM.:w
	if (!decode){
		for (int i = 0; i < recur_output_blobs_.size(); ++i) {
			caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
				recur_output_blobs_[i]->mutable_cpu_diff());
		}
	}
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  CHECK_EQ(top.size(), output_blobs_.size());
	CHECK_GE(top.size(), output_blobs_.size());
//	for (int i = 0; i < top.size(); ++i) {
	for (int i = 0; i < output_blobs_.size(); ++i) {
		top[i]->ReshapeLike(*output_blobs_[i]);
		top[i]->ShareData(*output_blobs_[i]);
		top[i]->ShareDiff(*output_blobs_[i]);
	}

	//added by xu shen
	//output all c_Ts and h_Ts
	if (top.size() > output_blobs_.size()){
		top[output_blobs_.size()] = all_end_seq_h_;
		top[output_blobs_.size() + 1] = all_end_seq_c_;
	}

	x_input_blob_->ShareData(*bottom[0]);
	x_input_blob_->ShareDiff(*bottom[0]);
	cont_input_blob_->ShareData(*bottom[1]);
	if (static_input_) {
		x_static_input_blob_->ShareData(*bottom[2]);
		x_static_input_blob_->ShareDiff(*bottom[2]);
	}
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Reset() {
	// "Reset" the hidden state of the net by zeroing out all recurrent outputs.
	for (int i = 0; i < recur_output_blobs_.size(); ++i) {
		caffe_set(recur_output_blobs_[i]->count(), Dtype(0),
			recur_output_blobs_[i]->mutable_cpu_data());
	}
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Hacky fix for test time... reshare all the shared blobs.
	// TODO: somehow make this work non-hackily.
	if (this->phase_ == TEST) {
		unrolled_net_->ShareWeightData();
	}

	//before the forward pass, we copy the last c and last h 
	//to c_0 and h_0
	DCHECK_EQ(recur_input_blobs_.size(), recur_output_blobs_.size());
	for (int i = 0; i < recur_input_blobs_.size(); ++i) {
		const int count = recur_input_blobs_[i]->count();
		DCHECK_EQ(count, recur_output_blobs_[i]->count());
		const Dtype* timestep_T_data = recur_output_blobs_[i]->cpu_data();
		Dtype* timestep_0_data = recur_input_blobs_[i]->mutable_cpu_data();
		//from time T to time 0
		caffe_copy(count, timestep_T_data, timestep_0_data);
	}

	unrolled_net_->ForwardPrefilled();
}

template <typename Dtype>
void RecurrentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";
	//if (!propagate_down[0] && !propagate_down[2]) { LOG(INFO) << "NOT BP"; return; }
	//TODO: skip backpropagation to inputs and parameters inside the unrolled
	//net according to propagate_down[0] and propagate_down[2]. For now just
	//backprop to inputs and parameters unconditionally, as either the inputs or 
	//the parameters do need backward(or Net would have set layer_needs_backward_[i]==false
	//for this layer).
	unrolled_net_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RecurrentLayer, Forward);
#endif

INSTANTIATE_CLASS(RecurrentLayer);

}  // namespace caffe
