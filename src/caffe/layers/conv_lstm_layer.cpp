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
void ConvLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void ConvLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + this->int_to_str(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void ConvLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void ConvLSTMLayer<Dtype>::ConcatSeqEndBlobNames(vector<string>* names) const {
	names->resize(2);
	(*names)[0] = "h_T_concat";
	(*names)[1] = "c_T_concat";
}

template <typename Dtype>
void ConvLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.conv_recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.conv_recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.conv_recurrent_param().bias_filler();

  const int seq_len = this->layer_param_.conv_recurrent_param().sequence_length();

  ConvRecurrentParameter conv_recurr_param = this->layer_param_.conv_recurrent_param();
 

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  // we use independent layer for i, f, o, and g
  LayerParameter hidden_param;
  hidden_param.set_type("Convolution");
  hidden_param.mutable_convolution_param()->CopyFrom(conv_recurr_param);
  hidden_param.mutable_convolution_param()->set_bias_term(false);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_convolution_param()->set_bias_term(true);
  biased_hidden_param.mutable_convolution_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  LayerParameter split_param;
  split_param.set_type("Split");

  BlobShape input_shape;
  input_shape.add_dim(1);  // c_0 and h_0 are a single timestep
  input_shape.add_dim(this->channels_);
  input_shape.add_dim(this->height_);
  input_shape.add_dim(this->width_);

  net_param->add_input("c_0");
  net_param->add_input_shape()->CopyFrom(input_shape);

  net_param->add_input("h_0");
  net_param->add_input_shape()->CopyFrom(input_shape);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
  }

  if (this->static_input_) {
    // Add layer to transform x_static to the gate dimension.
    //     W_xc_x_static = W_xc_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xc_x_static");
    x_static_transform_param->add_param()->set_name("W_xc_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xc_x_static");

    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    new_shape->add_dim(this->N_);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->add_bottom("W_xc_x_static");
    reshape_param->add_top("W_xc_x_static");
  }

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = this->int_to_str(t - 1);
    string ts = this->int_to_str(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xc_x_" + ts);

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(sum_param);
      cont_h_param->mutable_eltwise_param()->set_coeff_blob(true);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static");
      }
      input_sum_layer->add_top("gate_input_" + ts);
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("cont_" + ts);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  {
	  LayerParameter* c_T_copy_param = net_param->add_layer();
	  c_T_copy_param->CopyFrom(split_param);
	  c_T_copy_param->add_bottom("c_" + this->int_to_str(this->T_));
	  c_T_copy_param->add_top("c_T");
  }

  if (decode_){
	  int num_seq = this->T_ / seq_len;

	  LayerParameter* h_T_param = net_param->add_layer();
	  h_T_param->add_top("h_T_concat");
	  h_T_param->set_name("h_T_cont");
	  h_T_param->set_type("Concat");
	  h_T_param->mutable_concat_param()->set_axis(0);

	  LayerParameter* c_T_param = net_param->add_layer();
	  c_T_param->add_top("c_T_concat");
	  c_T_param->set_name("c_T_cont");
	  c_T_param->set_type("Concat");
	  c_T_param->mutable_concat_param()->set_axis(0);

	  for (int n = 0; n < num_seq; n++){
		  h_T_param->add_bottom("h_" + this->int_to_str(n * seq_len + seq_len));
		  c_T_param->add_bottom("c_" + this->int_to_str(n * seq_len + seq_len));
	  }
  }

  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(ConvLSTMLayer);
REGISTER_LAYER_CLASS(ConvLSTM);

}  // namespace caffe
