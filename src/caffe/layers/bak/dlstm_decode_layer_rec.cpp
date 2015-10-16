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
		names->resize(2);
		(*names)[0] = "h_mem";
		(*names)[1] = "h_tut";
	}

	//TODO: split the process of encoding and decoding
	template <typename Dtype>
	void DLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const{
		const int num_output = this->layer_param_.recurrent_param().num_output();
		const int sequence_len = this->layer_param_.recurrent_param().sequence_length();
		CHECK_GT(sequence_len, 0) << "sequence length must be positive";
		CHECK_GT(num_output, 0) << "num_output must be positive";
		CHECK(this->T_ % sequence_len == 0) << 
			"num of samples should be equal to batch_size * sequence_length.";
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
		hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
		hidden_param.mutable_inner_product_param()->set_bias_term(false);
		hidden_param.mutable_inner_product_param()->set_axis(2);
		hidden_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(weight_filler);
		//bias
		LayerParameter biased_hidden_param(hidden_param);
		biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
		biased_hidden_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(bias_filler);

		//Add layer to transform all timesteps of h_rec to the x dimension
		//in decoding LSTM.
		//this is useful in testing stage
//		LayerParameter biased_dec_trans_param;
//		biased_dec_trans_param.set_type("InnerProduct");
//		biased_dec_trans_param.mutable_inner_product_param()->set_num_output(num_rec_feature);
//		biased_dec_trans_param.mutable_inner_product_param()->set_bias_term(true);
//		biased_dec_trans_param.mutable_inner_product_param()->set_axis(2);
//		biased_dec_trans_param.mutable_inner_product_param()->mutable_weight_filler()->CopyFrom(dec_trans_weight_filler);
//		biased_dec_trans_param.mutable_inner_product_param()->mutable_bias_filler()->CopyFrom(dec_trans_bias_filler);

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

		//contencate in dim 1
		//slice in dim 0
		LayerParameter* cont_slice_param = net_param->add_layer();
		cont_slice_param->CopyFrom(slice_param);
		cont_slice_param->set_name("cont_slice");
		cont_slice_param->add_bottom("cont");
		cont_slice_param->mutable_slice_param()->set_axis(1);

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

		//Add layer to transform all timesteps of x to the hidden state dimension
		//in decoder LSTM.
		//W_xc_x_d = W_xc_d * x + b_c_d
		{
			//missing type
			LayerParameter* x_transform_param = net_param->add_layer();
			x_transform_param->CopyFrom(biased_hidden_param);
			x_transform_param->set_name("x_transform_d");
			//param: share weights and biases
			x_transform_param->add_param()->set_name("W_xc_d");
			x_transform_param->add_param()->set_name("b_c_d");
			x_transform_param->add_bottom("x");
			x_transform_param->add_top("W_xc_x_d");
		}


		if (this->static_input_){
			//Add layer to transform x_static to the gate dimension.
			//      W_xc_x_static = W_xc_static * x_static
			//static input no bias
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

		//slice input along axis 0(time), to get each time step input
		//default slice step is 1 along given axis
		LayerParameter* x_slice_param_d = net_param->add_layer();
		x_slice_param_d->CopyFrom(slice_param);
		x_slice_param_d->add_bottom("W_xc_x_d");
		x_slice_param_d->set_name("W_xc_x_slice_d");

		LayerParameter output_concat_layer_mem;
		output_concat_layer_mem.set_name("h_concat_mem");
		output_concat_layer_mem.set_type("Concat");
		output_concat_layer_mem.add_top("h_mem");
		//concatenate along axis 0(time)
		output_concat_layer_mem.mutable_concat_param()->set_axis(0);

		LayerParameter output_concat_layer_tut;
		output_concat_layer_tut.set_name("h_concat_tut");
		output_concat_layer_tut.set_type("Concat");
		output_concat_layer_tut.add_top("h_tut");
		//concatenate along axis 0(time)
		output_concat_layer_tut.mutable_concat_param()->set_axis(0);

		//save parameter for every time step
		for (int t = 1; t <= this->T_; ++t){
			string tm1s = this->int_to_str(t - 1);
			string ts = this->int_to_str(t);

			cont_slice_param->add_top("cont_" + ts);
			x_slice_param->add_top("W_xc_x_" + ts);

			//Add layers to flush the hidden state when beginning a new 
			//sequence, as indicated by cont_t.
			//		h_conted_{t-1} := cont_t * h_{t-1}
			//
			//    Normally, cont_t is binary (i.e., 0 or 1), so:
			//      h_conted_{t-1} := h_{t-1} if cont_t == 1
			//								  0 otherwise
			{
				LayerParameter* cont_h_param = net_param->add_layer();
				cont_h_param->CopyFrom(sum_param);
				cont_h_param->mutable_eltwise_param()->set_coeff_blob(true);
				cont_h_param->set_name("h_conted_" + tm1s);
				cont_h_param->add_bottom("h_" + tm1s);
				cont_h_param->add_bottom("cont_" + ts);
				cont_h_param->add_top("h_conted_" + tm1s);
			}

			//Add layer to compute
			//   W_hc_h_{t-1} := W_hc * h_conted_{t-1}
			{
				LayerParameter* w_param = net_param->add_layer();
				w_param->CopyFrom(hidden_param);
				w_param->set_name("transform_" + ts);
				w_param->add_param()->set_name("W_hc");
				w_param->add_bottom("h_conted_" + tm1s);
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

		//slice input X by to x_t
		for (int dt = 1; dt <= this->T_; ++dt){
			x_slice_param_d->add_top("W_xc_x_d_" + this->int_to_str(dt));
		}

		//record the corresponding decoding hidden layer names
		vector<string> not_used_c_names;
		//set up decoding LSTM
		for (int dt = 1; dt <= this->T_; ++dt){
			string dts = this->int_to_str(dt + this->T_);
			string dtm1s = this->int_to_str(dt + this->T_ - 1);
			string ots = this->int_to_str(dt);
			//the ending time of corresponding encoding sequence
			int seq_reversed_t = dt % sequence_len == 0 ? dt - sequence_len + 1 :
				(dt / sequence_len) * sequence_len +
				(sequence_len - (dt % sequence_len) + 1);
			int dtm1 = dt - 1;
			int seq_reversed_tm1 = dtm1 % sequence_len == 0 ? dtm1 - sequence_len + 1 :
				(dtm1 / sequence_len) * sequence_len +
				(sequence_len - (dtm1 % sequence_len) + 1);
			string seq_reversed_tm1s = this->int_to_str(seq_reversed_tm1);
			string seq_reversed_ts = this->int_to_str(seq_reversed_t);
			string dec_h_name = (dt - 1) % sequence_len == 0 ? "h_" + seq_reversed_ts :
				"h_" + dtm1s;
			string dec_c_name = (dt - 1) % sequence_len == 0 ? "c_" + seq_reversed_ts :
				"c_" + dtm1s;

			//Add Layer to transform h_dec as input
			//h_dec_transform_t = W_xc_d_t * h_dec + b_d
			/*
			{
				LayerParameter* h_dec_transform_param = net_param->add_layer();
				h_dec_transform_param->CopyFrom(biased_dec_trans_param);
				h_dec_transform_param->set_name("h_dec_transform_" + ots);
				h_dec_transform_param->mutable_inner_product_param()->set_num_output(4 * num_output);
				h_dec_transform_param->mutable_inner_product_param()->set_axis(2);
				h_dec_transform_param->add_param()->set_name("W_xc_d");
				h_dec_transform_param->add_param()->set_name("b_c_d");
				h_dec_transform_param->add_bottom(dec_h_name);
				h_dec_transform_param->add_top("h_dec_transform_" + ots);
			}
			*/
			
			//Add layer to compute
			//   W_hc_h_{t-1} := W_hc * h_conted_{t-1}
			{
				LayerParameter* dw_param = net_param->add_layer();
				dw_param->CopyFrom(hidden_param);
				dw_param->set_name("transform_d_" + dts);
				//use different decoding weight here, need to train independently
				dw_param->add_param()->set_name("W_hc_d");
				//here we don't need to use conted h, because in beginning of a sequence, 
				//we just copy encoding end h_T as starting h
				dw_param->add_bottom(dec_h_name);
				dw_param->add_top("W_hc_h_" + dtm1s);
				//sum along streams and times
				dw_param->mutable_inner_product_param()->set_axis(2);
			}


			//Add the outputs of the linear transformations to compute the gate input.
			//      gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
			//                    = W_hc_h_{t-1} + W_xc_x_t + b_C
			{
				//the beginning of decoding sequence has no input
				if ((dt - 1) % sequence_len == 0){
					LayerParameter* dinput_sum_layer = net_param->add_layer();
					dinput_sum_layer->set_name("gate_input_" + dts);
					dinput_sum_layer->add_bottom("W_hc_h_" + dtm1s);
					dinput_sum_layer->set_type("Copy");
					dinput_sum_layer->add_top("gate_input_" + dts);
				}
				else{
					LayerParameter* dinput_sum_layer = net_param->add_layer();
					dinput_sum_layer->CopyFrom(sum_param);
					dinput_sum_layer->set_name("gate_input_" + dts);
					dinput_sum_layer->add_bottom("W_hc_h_" + dtm1s);
					dinput_sum_layer->add_bottom("W_xc_x_d_" + seq_reversed_tm1s);
					dinput_sum_layer->add_top("gate_input_" + dts);
				}
			}

			//Add the LSTM Unit to learn sequence feature
			{
				LayerParameter* dlstm_unit_param = net_param->add_layer();
				//in decoding stage, set decode to be true
				dlstm_unit_param->set_type("LSTMUnit");
				dlstm_unit_param->add_bottom(dec_c_name);
				dlstm_unit_param->add_bottom("gate_input_" + dts);
				dlstm_unit_param->add_bottom("cont_" + ots);
				dlstm_unit_param->add_top("c_" + dts);
				dlstm_unit_param->add_top("h_" + dts);
				dlstm_unit_param->mutable_recurrent_param()->set_decode(true);
				dlstm_unit_param->set_name("unit_" + dts);
				if (dt % sequence_len == 0){
					not_used_c_names.push_back("c_" + dts);
				}
			}
		}// for (int dt = 1; dt <= this->T_; ++dt)


		//add a silence layer to silence the not used h_dec_ts and c_dec_ts
		{
			LayerParameter* silence_param = net_param->add_layer();
			silence_param->set_type("Silence");
			silence_param->set_name("silence_cs");
			for (int i = 0; i < not_used_c_names.size(); i++){
				silence_param->add_bottom(not_used_c_names[i]);
			}
		}

		//in the reconstruction model, the reconstruction order is reversed
		{
			const int n_seqs = this->T_ / sequence_len;
			for (int n = 0; n < n_seqs; n++){
				for (int s = sequence_len; s >= 1; s--){
					string rec_ts = "h_" + this->int_to_str(this->T_ + n * sequence_len + s);
					output_concat_layer_mem.add_bottom(rec_ts);
				}
			}
		}

		{
			//it seems that no name here
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
