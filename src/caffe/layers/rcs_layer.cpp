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
string RCSLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void RCSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
	  << "bottom[0] must have at least 2 axes -- (#VideoNum, #category, ...)";
  
  cate_axis_ = this->layer_param_.rcs_param().cate_axis();
  C_ = bottom[0]->shape(cate_axis_);
  LOG(INFO) << "Initializing RCS layer: contains "
	  << C_ << " categories.";

  // Create a NetParameter; setup the inputs that aren't unique to particular 
  // RCS architectures
  NetParameter net_param;
  net_param.set_force_backward(true);

  net_param.add_input("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  net_param.add_input_shape()->CopyFrom(input_shape);

  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // RCS architecture.
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
      this->layer_param_.rcs_param().debug_info());

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
void RCSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  x_input_blob_->ShareData(*bottom[0]);
  x_input_blob_->ShareDiff(*bottom[0]);

  top[0]->ReshapeLike(*x_output_blob_);
  x_output_blob_->ShareData(*top[0]);
  x_output_blob_->ShareDiff(*top[0]);
  //top[0]->ShareData(*x_output_blob_);
  //top[0]->ShareDiff(*x_output_blob_);
}

template<typename Dtype>
void RCSLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const
{
	const FillerParameter& weight_filler = this->layer_param_.rcs_param().weight_filler();
	const FillerParameter& bias_filler = this->layer_param_.rcs_param().bias_filler();

	// Add generic LayerParameter's (without bottoms/tops) of layer types we'll
	// use to save redundant code.
	// 1. split layer
	/*LayerParameter split_param;
	split_param.set_type("Split");

	LayerParameter* cate_split_param = net_param->add_layer();
	cate_split_param->CopyFrom(split_param);
	cate_split_param->set_name("cate_split");
	cate_split_param->add_bottom("x");*/

	// 2. part_sort layer
	LayerParameter part_sort_param;
	part_sort_param.set_type("PartSort");
	part_sort_param.mutable_part_sort_param()->set_axis(1);

	// 3. constrain ip layer
	LayerParameter ip_param;
	ip_param.set_type("InnerProduct");
	ip_param.mutable_inner_product_param()->set_num_output(1); // every ip_layer in RCS has only 1 output
	ip_param.mutable_inner_product_param()->set_bias_term(true);
	ip_param.mutable_inner_product_param()->set_axis(1);
	ip_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(weight_filler);
	ip_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(bias_filler);

	// 4. concat layer
	LayerParameter output_concat_layer;
	output_concat_layer.set_name("o_concat");
	output_concat_layer.set_type("Concat");
	output_concat_layer.add_top("o");
	output_concat_layer.mutable_concat_param()->set_axis(1);


	//network scaffolding
	for (int c = 0; c < C_; c++)
	{
		string cs = this->int_to_str(c);

		//1. split
		//cate_split_param->add_top("x_c" + cs);

		//2. part_sort
		LayerParameter* x_part_sort_layer_param = net_param->add_layer();
		x_part_sort_layer_param->CopyFrom(part_sort_param);
		x_part_sort_layer_param->mutable_part_sort_param()->set_first_element(c);
		x_part_sort_layer_param->set_name("x_c" + cs + "_part_sort");
		x_part_sort_layer_param->add_bottom("x");
		//x_part_sort_layer_param->add_bottom("x_c" + cs);
		x_part_sort_layer_param->add_top("x_c" + cs + "_part_sort");

		//3. constrain ip
		LayerParameter* x_ip_layer_param = net_param->add_layer();
		x_ip_layer_param->CopyFrom(ip_param);
		x_ip_layer_param->set_name("x_c" + cs + "_ip");


		//x_ip_layer_param->add_param()->set_name("W_c" + cs);
		//x_ip_layer_param->add_param()->set_name("b_c" + cs);
		/*x_ip_layer_param->mutable_param(0)->set_lr_mult(500);
		x_ip_layer_param->mutable_param(0)->set_decay_mult(1);
		x_ip_layer_param->mutable_param(1)->set_lr_mult(500*2);
		x_ip_layer_param->mutable_param(1)->set_decay_mult(0);*/


		x_ip_layer_param->add_bottom("x_c" + cs + "_part_sort");
		x_ip_layer_param->add_top("x_c" + cs + "_ip");

		//4. concate
		output_concat_layer.add_bottom("x_c" + cs + "_ip");
	}

	net_param->add_layer()->CopyFrom(output_concat_layer);
}

template <typename Dtype>
void RCSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeightData();
  }



  unrolled_net_->ForwardPrefilled();


}

template <typename Dtype>
void RCSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//if (!propagate_down[0]) { LOG(INFO) << "NOT BP"; return; }

  unrolled_net_->Backward();

	/*for (int i = 0; i < 5; i++)
	{
		LOG(INFO) << bottom[0]->cpu_diff()[i];
	}*/
}

#ifdef CPU_ONLY
STUB_GPU(RCSLayer);
#endif

INSTANTIATE_CLASS(RCSLayer);
REGISTER_LAYER_CLASS(RCS);

}  // namespace caffe
