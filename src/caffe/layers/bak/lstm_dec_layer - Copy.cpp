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
	void DLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h_0";
		(*names)[1] = "c_0";
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
		names->resize(2);
		(*names)[0] = "h_" + this->int_to_str(this->T_);
		(*names)[1] = "c_T";
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
		names->resize(1);
		(*names)[0] = "h";
	}

	//TODO: split the process of encoding and decoding
	template <typename Dtype>
	void DLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const{
		const int num_output = this->layer_param_.recurrent_param().num_output();
		const int sequence_len = this->layer_param_.recurrent_param().sequence_length();
		const bool pred = this->layer_param_.recurrent_param().pred();
		CHECK_GT(sequence_len, 0) << "sequence length must be positive";
		CHECK_GT(num_output, 0) << "num_output must be positive";
		CHECK(this->T_ % sequence_len == 0) << 
			"num of samples should be equal to batch_size * sequence_length.";
		//must share weights with LSTM layer
		const FillerParameter& weight_filler =
			this->layer_param_.recurrent_param().weight_filler();
		const FillerParameter& bias_filler =
			this->layer_param_.recurrent_param().bias_filler();
		const FillerParameter& dec_trans_weight_filler =
			this->layer_param_.recurrent_param().dec_trans_weight_filler();
		const FillerParameter& dec_trans_bias_filler =
			this->layer_param_.recurrent_param().dec_trans_bias_filler();
		//Add generic LayerParameter's (without bottoms/tops) of layer types we'll
		//use to save redundant code.
		LayerParameter hidden_param;
		hidden_param.set_type("InnerProduct");
		//i_t, f_t, o_t, g_t
		hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
		hidden_param.mutable_inner_product_param()->set_bias_term(false);
		hidden_param.mutable_inner_product_param()->set_axis(2);
		hidden_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(weight_filler);
		//bias
		LayerParameter biased_hidden_param(hidden_param);
		biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
		biased_hidden_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(bias_filler);

		//sum
		LayerParameter sum_param;
		sum_param.set_type("Eltwise");
		sum_param.mutable_eltwise_param()->set_operation(
			EltwiseParameter_EltwiseOp_SUM);

		//slice
		LayerParameter slice_param;
		slice_param.set_type("Slice");
		slice_param.mutable_slice_param()->set_axis(0);

		//split
		LayerParameter split_param;
		split_param.set_type("Split");

		BlobShape input_shape;
		input_shape.add_dim(1);//c_0 and h_0 are a single timestep
		//c and h are with the same dim
		input_shape.add_dim(this->N_);
		input_shape.add_dim(num_output);

		net_param->add_input("c_0");
		net_param->add_input_shape()->CopyFrom(input_shape);

		net_param->add_input("h_0");
		net_param->add_input_shape()->CopyFrom(input_shape);

		//slice h in dim 0
		LayerParameter* h_slice_param = net_param->add_layer();
		h_slice_param->CopyFrom(slice_param);
		h_slice_param->set_name("h_slice");
		h_slice_param->add_bottom("h_enc_T");
		h_slice_param->mutable_slice_param()->set_axis(0);

		LayerParameter* c_slice_param = net_param->add_layer();
		c_slice_param->CopyFrom(slice_param);
		c_slice_param->set_name("c_slice");
		c_slice_param->add_bottom("c_enc_T");
		c_slice_param->mutable_slice_param()->set_axis(0);


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


		//slice input along axis 0(time), to get each time step input
		//default slice step is 1 along given axis
		LayerParameter* x_slice_param = net_param->add_layer();
		x_slice_param->CopyFrom(slice_param);
		x_slice_param->add_bottom("W_xc_x");
		x_slice_param->set_name("W_xc_x_slice");

		//slice input along axis 0(time), to get each time step input
		//default slice step is 1 along given axis
		LayerParameter* x_slice_param_d = net_param->add_layer();
		x_slice_param_d->CopyFrom(slice_param);
		x_slice_param_d->add_bottom("W_xc_x_d");
		x_slice_param_d->set_name("W_xc_x_slice_d");

		LayerParameter output_concat_layer_mem;
		output_concat_layer_mem.set_name("h_concat");
		output_concat_layer_mem.set_type("Concat");
		output_concat_layer_mem.add_top("h");
		//concatenate along axis 0(time)
		output_concat_layer_mem.mutable_concat_param()->set_axis(0);


		//save parameter for every time step
		for (int t = 1; t <= this->T_; ++t){
			string tm1s = this->int_to_str(t - 1);
			string ts = this->int_to_str(t);

			cont_slice_param->add_top("cont_" + ts);
			x_slice_param->add_top("W_xc_x_" + ts);

			//Add layer to compute
			// W_hc_h_{t-1} := W_hc * h_{t-1}
			// we don't need cont variables in decoding LSTM
			{
				LayerParameter* w_param = net_param->add_layer();
				w_param->CopyFrom(hidden_param);
				w_param->set_name("transform_" + ts);
				w_param->add_param()->set_name("W_hc");
				w_param->add_bottom("h_" + tm1s);
				w_param->add_top("W_hc_h_" + tm1s);
				//sum along streams and times
				w_param->mutable_inner_product_param()->set_axis(2);
			}

			//Add the outputs of the linear transformations to compute the gate input.
			//      gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
			//                    = W_hc_h_{t-1} + W_xc_x_t + b_C
			{
				LayerParameter* input_sum_layer = net_param->add_layer();
				input_sum_layer->CopyFrom(sum_param);
				input_sum_layer->set_name("gate_input_" + ts);
				input_sum_layer->add_bottom("W_hc_h_" + tm1s);
				input_sum_layer->add_bottom("W_xc_x_" + ts);
				if (this->static_input_){
					input_sum_layer->add_bottom("W_xc_x_static");
				}
				input_sum_layer->add_top("gate_input_" + ts);
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
				LayerParameter* lstm_unit_param = net_param->add_layer();
				lstm_unit_param->set_type("LSTMUnit");
				lstm_unit_param->add_bottom("c_" + tm1s);
				lstm_unit_param->add_bottom("gate_input_" + ts);
				lstm_unit_param->add_bottom("cont_" + ts);
				lstm_unit_param->add_top("c_" + ts);
				lstm_unit_param->add_top("h_" + ts);
				lstm_unit_param->set_name("unit_" + ts);
			}

			output_concat_layer_tut.add_bottom("h_" + ts);
		}//for (int t =1; t <= this->T_; ++t)

		{
			LayerParameter* c_T_copy_param = net_param->add_layer();
			c_T_copy_param->CopyFrom(split_param);
			c_T_copy_param->add_bottom("c_" + this->int_to_str(this->T_));
			c_T_copy_param->add_top("c_T");
		}

		net_param->add_layer()->CopyFrom(output_concat_layer_tut);
		net_param->add_layer()->CopyFrom(output_concat_layer_mem);
	}

	INSTANTIATE_CLASS(DLSTMLayer);
	REGISTER_LAYER_CLASS(DLSTM);
}
