#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
string SSDLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void SSDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 3)
	  << "bottom[0] must have at least 3 axes -- (#VideoNum, #timesteps, #category, ...)";
  
  cate_axis_ = this->layer_param_.ssd_param().cate_axis();
  C_ = bottom[0]->shape(cate_axis_);
  constrain_ip_sum1_rate_ = this->layer_param_.ssd_param().constrain_ip_sum1_rate();
  constrain_ip_monotonic_rate_ = this->layer_param_.ssd_param().constrain_ip_monotonic_rate();
  LOG(INFO) << "Initializing SSD layer: contains "
	  << C_ << " categories.";

  // Create a NetParameter; setup the inputs that aren't unique to particular 
  // SSD architectures
  NetParameter net_param;
  net_param.set_force_backward(true);

  net_param.add_input("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  net_param.add_input_shape()->CopyFrom(input_shape);


  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // SSD architecture.
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
  unrolled_net_.reset(new Net<Dtype>(net_param));
  unrolled_net_->set_debug_info(
      this->layer_param_.ssd_param().debug_info());

  // Setup pointers to the inputs and outputs.
  x_input_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("x").get());
  x_output_blob_ = CHECK_NOTNULL(unrolled_net_->blob_by_name("o").get());

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

}

template <typename Dtype>
void SSDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);

  top[0]->ReshapeLike(*x_output_blob_);
  //x_output_blob_->ShareData(*top[0]);
  //x_output_blob_->ShareDiff(*top[0]);
  top[0]->ShareData(*x_output_blob_);
  top[0]->ShareDiff(*x_output_blob_);
}

template<typename Dtype>
void SSDLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const
{
	const FillerParameter& weight_filler = this->layer_param_.ssd_param().weight_filler();
	const FillerParameter& bias_filler = this->layer_param_.ssd_param().bias_filler();

	// Add generic LayerParameter's (without bottoms/tops) of layer types we'll
	// use to save redundant code.
	// 1. slice layer
	LayerParameter slice_param;
	slice_param.set_type("Slice");
	slice_param.mutable_slice_param()->set_axis(cate_axis_);

	LayerParameter* cate_slice_param = net_param->add_layer();
	cate_slice_param->CopyFrom(slice_param);
	cate_slice_param->set_name("cate_slice");
	cate_slice_param->add_bottom("x");

	// 2. sort layer
	LayerParameter sort_param;
	sort_param.set_type("Sort");
	sort_param.mutable_sort_param()->set_axis(1);

	// 3. constrain ip layer
	LayerParameter constrain_ip_param;
	constrain_ip_param.set_type("ConstrainIP");
	constrain_ip_param.mutable_constrain_ip_param()->set_num_output(1); // every constrain_ip_layer in ssd has only 1 output
	constrain_ip_param.mutable_constrain_ip_param()->set_bias_term(true);
	constrain_ip_param.mutable_constrain_ip_param()->set_axis(1);
	constrain_ip_param.mutable_constrain_ip_param()->set_sum1_rate(constrain_ip_sum1_rate_);
	constrain_ip_param.mutable_constrain_ip_param()->set_monotonic_rate(constrain_ip_monotonic_rate_);
	constrain_ip_param.mutable_constrain_ip_param()->mutable_weight_filler()->CopyFrom(weight_filler);
	constrain_ip_param.mutable_constrain_ip_param()->mutable_bias_filler()->CopyFrom(bias_filler);

	// 4. concat layer
	LayerParameter output_concat_layer;
	output_concat_layer.set_name("o_concat");
	output_concat_layer.set_type("Concat");
	output_concat_layer.add_top("o");
	output_concat_layer.mutable_concat_param()->set_axis(1);


	//network scaffolding
	for (int c = 1; c <= this->C_; c++)
	{
		string cs = this->int_to_str(c);

		//1. slice
		cate_slice_param->add_top("x_c" + cs);

		//2. sort
		LayerParameter* x_sort_layer_param = net_param->add_layer();
		x_sort_layer_param->CopyFrom(sort_param);
		x_sort_layer_param->set_name("x_c" + cs + "_sort");
		x_sort_layer_param->add_bottom("x_c" + cs);
		x_sort_layer_param->add_top("x_c" + cs + "_sort");

		//3. constrain ip
		LayerParameter* x_constrain_ip_layer_param = net_param->add_layer();
		x_constrain_ip_layer_param->CopyFrom(constrain_ip_param);
		x_constrain_ip_layer_param->set_name("x_c" + cs + "_constrain_ip");

		x_constrain_ip_layer_param->add_param()->set_name("W_c" + cs);
		x_constrain_ip_layer_param->add_param()->set_name("b_c" + cs);

		x_constrain_ip_layer_param->add_bottom("x_c" + cs + "_sort");
		x_constrain_ip_layer_param->add_top("x_c" + cs + "_constrain_ip");

		//4. concate
		output_concat_layer.add_bottom("x_c" + cs + "_constrain_ip");
	}

	net_param->add_layer()->CopyFrom(output_concat_layer);
}

template <typename Dtype>
void SSDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeightData();
  }

  unrolled_net_->ForwardPrefilled();
}

template <typename Dtype>
void SSDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//if (!propagate_down[0]) { LOG(INFO) << "NOT BP"; return; }
  unrolled_net_->Backward();
}

#ifdef CPU_ONLY
STUB_GPU(SSDLayer);
#endif

INSTANTIATE_CLASS(SSDLayer);
REGISTER_LAYER_CLASS(SSD);

}  // namespace caffe
