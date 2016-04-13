/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/10/4
** desc: DecodingLSTMLayer(CPU)
** TODO: here I just deal with fixed sequence length decoding
**       we need to extend it to deal with various length
*********************************************************************************/
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

	template <typename Dtype>
	void DLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
		names->resize(1);
		(*names)[0] = "h_dec";
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const{
		const int num_output = this->layer_param_.recurrent_param().num_output();
		const int seq_len = this->layer_param_.recurrent_param().sequence_length();
		const bool reverse = this->layer_param_.recurrent_param().reverse();
		CHECK_GT(seq_len, 0) << "sequence length must be positive";
		CHECK(this->T_ % seq_len == 0) << "T_ must be divided by sequence length";
		CHECK_GT(num_output, 0) << "num_output must be positive";
		//must share weights with LSTM layer
		const FillerParameter& weight_filler =
			this->layer_param_.recurrent_param().weight_filler();
		const FillerParameter& bias_filler =
			this->layer_param_.recurrent_param().bias_filler();
		//Add generic LayerParameter's (without bottoms/tops) of layer types we'll
		//use to save redundant code.
		LayerParameter hidden_param;
		hidden_param.set_type("InnerProduct");

		//i_t, f_t, o_t, g_t
		//no bias here?
		hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
		hidden_param.mutable_inner_product_param()->set_bias_term(false);
		hidden_param.mutable_inner_product_param()->set_axis(2);
		hidden_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(weight_filler);

		//biased param
		LayerParameter biased_hidden_param(hidden_param);
		biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
		biased_hidden_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(bias_filler);

		//sum
		LayerParameter sum_param;
		sum_param.set_type("Eltwise");
		sum_param.mutable_eltwise_param()->set_operation(
			EltwiseParameter_EltwiseOp_SUM);

		LayerParameter slice_param;
		slice_param.set_type("Slice");
		slice_param.mutable_slice_param()->set_axis(0);

		LayerParameter split_param;
		split_param.set_type("Split");

		LayerParameter* h_slice_param = net_param->add_layer();
		h_slice_param->CopyFrom(slice_param);
		h_slice_param->set_name("h_slice");
		h_slice_param->add_bottom("h");

		LayerParameter* c_slice_param = net_param->add_layer();
		c_slice_param->CopyFrom(slice_param);
		c_slice_param->set_name("c_slice");
		c_slice_param->add_bottom("c");

		//TODO: Here we should also try to use the predictions of last step
		//      instead of ground truth input
		//      maybe in pretraining, precise information is better
		//Add layer to transform all timesteps of x to the hidden state dimension.
		//W_xc_x = W_xc * x + b_c
		{
			//missing type
			LayerParameter* x_transform_param = net_param->add_layer();
			x_transform_param->CopyFrom(biased_hidden_param);
			x_transform_param->set_name("x_transform");
			//param: share weights and biases
			x_transform_param->add_param()->set_name("W_xc");
			x_transform_param->add_param()->set_name("b_c");
			x_transform_param->add_bottom("x");
			x_transform_param->add_top("W_xc_x");
		}

		if (this->static_input_){
			//Add layer to transform x_static to the gate dimension.
			//      W_xc_x_static = W_xc_static * x_static
			LayerParameter* x_static_transform_param = net_param->add_layer();
			x_static_transform_param->CopyFrom(hidden_param);
			//lump from dim 1 into output of inner product
			x_static_transform_param->mutable_inner_product_param()->set_axis(1);
			x_static_transform_param->set_name("W_xc_x_static");
			x_static_transform_param->add_param()->set_name("W_xc_static");
			x_static_transform_param->add_bottom("x_static");
			x_static_transform_param->add_top("W_xc_x_static");

			LayerParameter* reshape_param = net_param->add_layer();
			reshape_param->set_type("Reshape");
			BlobShape* new_shape =
				reshape_param->mutable_reshape_param()->mutable_shape();
			new_shape->add_dim(1); //one timestep
			new_shape->add_dim(this->N_);
			new_shape->add_dim(
				x_static_transform_param->inner_product_param().num_output());
			reshape_param->add_bottom("W_xc_x_static");
			reshape_param->add_top("W_xc_x_static");
		}

		//slice input along axis 0(time), to get each time step input
		//default slice step is 1 along given axis
		LayerParameter* x_slice_param = net_param->add_layer();
		x_slice_param->CopyFrom(slice_param);
		x_slice_param->add_bottom("W_xc_x");
		x_slice_param->set_name("W_xc_x_slice");

		int num_seq = this->T_ / seq_len;
		for (int n = 0; n < num_seq; n++){
			string ns = this->int_to_str(n);
			h_slice_param->add_top("h_" + ns + "_0");
			c_slice_param->add_top("c_" + ns + "_0");
		}

		/* if reverse, reverse input x in within sequence;
		 * else, just input x in original sequence order
		 */
		if (reverse){
			for (int n = 0; n < num_seq; n++){
				for (int s = seq_len; s >= 1; s--){
					string ns = this->int_to_str(n);
					string ss = this->int_to_str(s);
					x_slice_param->add_top("W_xc_x_" + ns + "_" + ss);
				}
			}
		}
		else{
			for (int n = 0; n < num_seq; n++){
				for (int s = 1 ; s <= seq_len; s++){
					string ns = this->int_to_str(n);
					string ss = this->int_to_str(s);
					x_slice_param->add_top("W_xc_x_" + ns + "_" + ss);
				}
			}
		}

		LayerParameter output_concat_layer;
		output_concat_layer.set_name("h_concat");
		output_concat_layer.set_type("Concat");
		output_concat_layer.add_top("h_dec");
		//concatenate along axis 0(time)
		output_concat_layer.mutable_concat_param()->set_axis(0);

		//save parameter for every time step
		for (int n = 0; n < num_seq; n++){
			for (int s = 1; s <= seq_len; s++){
				string ns = this->int_to_str(n);
				string ss = this->int_to_str(s);
				string units = ns + "_" + ss;
				string unitm1s = ns + "_" + this->int_to_str(s - 1);
				//Add layer to compute
				//   W_hc_h_{t-1} := W_hc * h_{t-1}
				{
					LayerParameter* w_param = net_param->add_layer();
					w_param->CopyFrom(hidden_param);
					w_param->set_name("transform_" + units);
					w_param->add_param()->set_name("W_hc");
					w_param->add_bottom("h_" + unitm1s);
					w_param->add_top("W_hc_h_" + unitm1s);
					//sum along streams and times
					w_param->mutable_inner_product_param()->set_axis(2);
				}

				//Add the outputs of the linear transformations to compute the gate input.
				//      gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
				//                    = W_hc_h_{t-1} + W_xc_x_t + b_C
				//to do prediction, first input is zero
				{
					LayerParameter* input_sum_layer = net_param->add_layer();
					if (s == 1){
						//split layer: if only has one top, just copy
						//if have more than two output, copy data, accumulate diffs
						input_sum_layer->set_type("Split");
						input_sum_layer->set_name("gate_input_" + units);
						input_sum_layer->add_bottom("W_hc_h_" + unitm1s);
						input_sum_layer->add_top("gate_input_" + units);
					}
					else{
						input_sum_layer->CopyFrom(sum_param);
						input_sum_layer->set_name("gate_input_" + units);
						input_sum_layer->add_bottom("W_hc_h_" + unitm1s);
						input_sum_layer->add_bottom("W_xc_x_" + unitm1s);
						if (this->static_input_){
							input_sum_layer->add_bottom("W_xc_x_static");
						}
						input_sum_layer->add_top("gate_input_" + units);
					}
				}

				//Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
				// Inputs: c_{t-1}, gate_input_t = {i_t, f_t, o_t, g_t), cont_t
				// Outputs: c_t, h_t
				//		[ i_t']
				//		[ f_t']  := gate_input_t
				//		[ o_t'] 
				//		[ g_t']
				//				i_t := \sigmoid[i_t']
				//				f_t := \sigmoid[f_t']
				//				o_t := \sigmoid[o_t']
				//				g_t := \tanh[g_t']
				//				c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
				//				h_t := o_t .* \tanh[c_t]
				{
					LayerParameter* dlstm_unit_param = net_param->add_layer();
					dlstm_unit_param->set_type("DLSTMUnit");
					dlstm_unit_param->add_bottom("c_" + unitm1s);
					dlstm_unit_param->add_bottom("gate_input_" + units);
					dlstm_unit_param->add_top("c_" + units);
					dlstm_unit_param->add_top("h_" + units);
					dlstm_unit_param->set_name("unit_" + units);
				}
			}
		}

		/* if reverse, reverse output in within sequence
		 * else, just output h in original sequence order
		 */
		if (reverse){
			for (int n = 0; n < num_seq; n++){
				for (int s = seq_len; s >= 1; s--){
					string ss = this->int_to_str(s);
					string ns = this->int_to_str(n);
					string units = ns + "_" + ss;
					output_concat_layer.add_bottom("h_" + units);
				}
			}
		}
		else{
			for (int n = 0; n < num_seq; n++){
				for (int s = 1; s <= seq_len; s++){
					string ss = this->int_to_str(s);
					string ns = this->int_to_str(n);
					string units = ns + "_" + ss;
					output_concat_layer.add_bottom("h_" + units);
				}
			}
		}
		net_param->add_layer()->CopyFrom(output_concat_layer);
	}

	INSTANTIATE_CLASS(DLSTMLayer);
	REGISTER_LAYER_CLASS(DLSTM);
}
