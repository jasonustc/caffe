#include "caffe/layers/lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{

	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
		N_ = this->layer_param_.lstm_param().batch_size(); // batch size
		//number of hidden units
		num_hid_ = this->layer_param_.lstm_param().num_output();
		//input dimension
		input_dim_ = bottom[0]->count() / bottom[0]->num();

		//check if we need to set up weights
		if (this->blobs_.size() > 0){
			LOG(INFO) << "Skipping parameter initialization";
		}
		else{
			this->blobs_.resize(3);
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.lstm_param().weight_filler()));

			//input-to-gate weights
			//initialize the weights
			//TODO: compute the four weights indepently
			vector<int> weight_shape;
			weight_shape.push_back(4 * num_hid_);
			weight_shape.push_back(input_dim_);
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
			weight_filler->Fill(this->blobs_[0].get());

			// hidden-to-gate weights
			// Initialize the weight
			weight_shape.clear();
			weight_shape.push_back(4 * num_hid_);
			weight_shape.push_back(num_hid_);
			this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
			weight_filler->Fill(this->blobs_[1].get());

			// If necessary, initialize and fill the bias term
			vector<int> bias_shape(1, 4 * num_hid_);
			this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
				this->layer_param_.lstm_param().bias_filler()));
			bias_filler->Fill(this->blobs_[2].get());
		}//parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);

		vector<int> cell_shape;
		cell_shape.push_back(N_);
		cell_shape.push_back(num_hid_);
		c_0_.Reshape(cell_shape);
		h_0_.Reshape(cell_shape);
		c_T_.Reshape(cell_shape);
		h_T_.Reshape(cell_shape);
		h_to_h_.Reshape(cell_shape);

		vector<int> gate_shape;
		gate_shape.push_back(N_);
		gate_shape.push_back(4);
		gate_shape.push_back(num_hid_);
		h_to_gate_.Reshape(gate_shape);
	}

	/*
	 * Here N_ is the number of streams in input
	 * Generally, we canten input of different stream in the num dim to bottom[0]
	 * Then Here we compute each stream T_ / N_ independently 
	 */
	template <typename Dtype>
	void LSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// Figure out the dimensions
		T_ = bottom[0]->num() / N_; //length of sequence
		CHECK_EQ(bottom[0]->num() % N_, 0) << "Input size should "
			<< "be multiple of sequence length";
		CHECK_EQ(bottom[0]->count() / T_ / N_, input_dim_) <<
			"Input size incompatible with inner product parameters.";
		vector<int> original_top_shape;
		original_top_shape.push_back(T_ * N_);
		original_top_shape.push_back(num_hid_);
		top[0]->Reshape(original_top_shape);

		//gate initialization
		//here the order is a little bit wired
		vector<int> gate_shape;
		gate_shape.push_back(T_);
		gate_shape.push_back(N_);
		gate_shape.push_back(4);
		gate_shape.push_back(num_hid_);
		//since the four gate use the same activation function
		//so we can put them together
		pre_gate_.Reshape(gate_shape);
		gate_.Reshape(gate_shape);

		vector<int> top_shape;
		//single sequence first
		top_shape.push_back(T_);
		top_shape.push_back(N_);
		top_shape.push_back(num_hid_);
		cell_.Reshape(top_shape);
		top_.Reshape(top_shape);
		//output is just hidden states
		top_.ShareData(*top[0]);
		top_.ShareDiff(*top[0]);

		//setup the bias multiplier
		//use to efficiently get the matrix form of bias
		vector<int> multiplier_shape(1, N_ * T_);
		bias_multiplier_.Reshape(multiplier_shape);
		caffe_set(bias_multiplier_.count(), Dtype(1),
			bias_multiplier_.mutable_cpu_data());
	}

	template<typename Dtype>
	void LSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
		Dtype* top_data = top_.mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* clip = NULL;
		// bottom[1]: cont indication
		if (bottom.size() > 1){
			clip = bottom[1]->cpu_data();
			//NOTE: here I think check bottom[0]->num() is more reasonable
			CHECK_EQ(bottom[1]->num(), bottom[1]->count());
		}
		const Dtype* weight_i = this->blobs_[0]->cpu_data();
		const Dtype* weight_h = this->blobs_[1]->cpu_data();
		const Dtype* bias = this->blobs_[2]->cpu_data();
		Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
		Dtype* gate_data = gate_.mutable_cpu_data();
		Dtype* cell_data = cell_.mutable_cpu_data();
		//intermediate memory
		Dtype* h_to_gate = h_to_gate_.mutable_cpu_data();

		// Initialize previous state
		if (clip){
			//propagate from last batch to current batch
			caffe_copy(c_0_.count(), c_T_.cpu_data(), c_0_.mutable_cpu_data());
			caffe_copy(h_0_.count(), h_T_.cpu_data(), h_0_.mutable_cpu_data());
		}
		else{
			caffe_set(c_0_.count(), Dtype(0.), c_0_.mutable_cpu_data());
			caffe_set(h_0_.count(), Dtype(0.), h_0_.mutable_cpu_data());
		}

		//compute input to hidden forward propagation
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_ * N_, 4 * num_hid_, input_dim_,
			Dtype(1.), bottom_data, weight_i, Dtype(0.), pre_gate_data);
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_ * N_, 4 * num_hid_, 1,
			Dtype(1.), bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);

		//compute recurrent forward propagation
		for (int t = 0; t < T_; ++t){
			Dtype* h_t = top_data + top_.offset(t);
			Dtype* c_t = cell_data + cell_.offset(t);
			Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
			Dtype* gate_t = gate_data + gate_.offset(t);
			//h before activation
			Dtype* h_to_gate_t = h_to_gate;
			// clip_t == 0: beginning of sequence
			const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
			const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
			const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

			//hidden to hidden propagation
			//different streams have different outputs and the same weight
			caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4 * num_hid_, num_hid_,
				Dtype(1.), h_t_1, weight_h, Dtype(0.), h_to_gate);

			for (int n = 0; n < N_; ++n){
				//whether is the beginning of sequence
				//if no speicial clip is given, we just reset
				//h and c in the beginning of each sequence 
				const bool cont = clip_t ? clip_t[n] : t > 0;
				//the output of each stream is computed independently
				if (cont){
					caffe_add(4 * num_hid_, pre_gate_t, h_to_gate, pre_gate_t);
				}
				//Apply nonlinearity
				for (int d = 0; d < num_hid_; ++d){
					//i
					gate_t[d] = sigmoid(pre_gate_t[d]);
					//reset memory here
					//f
					gate_t[num_hid_ + d] = cont ? sigmoid(pre_gate_t[num_hid_ + d]) 
						: Dtype(0.);
					//o
					gate_t[2 * num_hid_ + d] = sigmoid(pre_gate_t[2 * num_hid_ + d]);
					//g
					gate_t[3 * num_hid_ + d] = tanh(pre_gate_t[3 * num_hid_ + d]);
					// Compute cell: c(t) = f(t)*c(t-1) + i(t)*g(t)
					c_t[d] = gate_t[num_hid_ + d] * c_t_1[d] + gate_t[d] *
						gate_t[3 * num_hid_ + d];
					h_t[d] = gate_t[2 * num_hid_ + d] * tanh(c_t[d]);
				}
				h_t += num_hid_;
				c_t += num_hid_;
				c_t_1 += num_hid_;
				pre_gate_t += 4 * num_hid_;
				gate_t += 4 * num_hid_;
				h_to_gate_t += 4 * num_hid_;
			}
		}
		//Preserve cell state and output value for trancated BPTT
		caffe_copy(N_ * num_hid_, cell_data + cell_.offset(T_ - 1), 
			c_T_.mutable_cpu_data());
		caffe_copy(N_ * num_hid_, top_data + top_.offset(T_ - 1),
			h_T_.mutable_cpu_data());
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const Dtype* top_data = top_.cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* clip = NULL;
		if (bottom.size() > 1){
			clip = bottom[1]->cpu_data();
			CHECK_EQ(bottom[0]->num(), bottom[1]->count()) << 
				"#cont elements should be equal to #sequences";
		}
		const Dtype* weight_i = this->blobs_[0]->cpu_data();
		const Dtype* weight_h = this->blobs_[1]->cpu_data();
		const Dtype* gate_data = gate_.cpu_data();
		const Dtype* cell_data = cell_.cpu_data();

		Dtype* top_diff = top_.mutable_cpu_diff();
		Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();
		Dtype* gate_diff = gate_.mutable_cpu_diff();
		Dtype* cell_diff = cell_.mutable_cpu_diff();

		//because top_ and h_T_ share the same data, so do not need to copy
		caffe_copy(N_ * num_hid_, c_T_.cpu_diff(), cell_diff + cell_.offset(T_ - 1));
		for (int t = T_ - 1; t >= 0; --t){
			Dtype* dh_t = top_diff + top_.offset(t);
			Dtype* dc_t = cell_diff + cell_.offset(t);
			Dtype* gate_diff_t = gate_diff + gate_.offset(t);
			Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
			Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t - 1) : 
				h_0_.mutable_cpu_diff();
			Dtype* dc_t_1 = t > 0 ? cell_diff + cell_.offset(t - 1) :
				c_0_.mutable_cpu_diff();
			const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
			const Dtype* c_t = cell_data + cell_.offset(t);
			const Dtype* c_t_1 = t > 0 ? cell_data + cell_.offset(t - 1) 
				: c_0_.cpu_data();
			const Dtype* gate_t = gate_data + gate_.offset(t);

			for (int n = 0; n < N_; ++n){
				const bool cont = clip_t ? clip_t[n] : t > 0;
				for (int d = 0; d < num_hid_; ++d){
					const Dtype tanh_c = tanh(c_t[d]);
					//o
					gate_diff_t[2 * num_hid_ + d] = dh_t[d] * tanh_c;
					//dtanh = 1 - tanh^2
					dc_t[d] += dh_t[d] * gate_t[2 * num_hid_ + d] * 
						(Dtype(1.) - tanh_c * tanh_c);
					dc_t_1[d] = cont ? dc_t[d] * gate_t[num_hid_ + d] : Dtype(0.);
					//f
					gate_diff_t[num_hid_ + d] = cont ? dc_t[d] * c_t_1[d] : Dtype(0.);
					//i
					gate_diff_t[d] = dc_t[d] * gate_t[3 * num_hid_ + d];
					//g
					gate_diff_t[3 * num_hid_ + d] = dc_t[d] * gate_t[d];

					pre_gate_diff_t[d] = gate_diff_t[d] * gate_t[d] * (1 - gate_t[d]);
					pre_gate_diff_t[num_hid_ + d] = gate_diff_t[num_hid_ + d] *
						gate_t[num_hid_ + d] * (1 - gate_t[num_hid_ + d]);
					pre_gate_diff_t[2 * num_hid_ + d] = gate_diff_t[2 * num_hid_ + d] *
						gate_t[2 * num_hid_ + d] * (1 - gate_t[2 * num_hid_ + d]);
					pre_gate_diff_t[3 * num_hid_ + d] = gate_diff_t[3 * num_hid_ + d] *
						gate_t[3 * num_hid_ + d] * (1 - gate_t[3 * num_hid_ + d]);
				}//for (int d = 0; d < num_hid_; ++d)
				// Clip derivatives before nonlinearity
				if (clipping_threshold_ > Dtype(0.)){
					caffe_bound(4 * num_hid_, pre_gate_diff_t, -clipping_threshold_,
						clipping_threshold_, pre_gate_diff_t);
				}
				dh_t += num_hid_;
				c_t += num_hid_;
				c_t_1 += num_hid_;
				dc_t += num_hid_;
				dc_t_1 += num_hid_;
				gate_t += 4 * num_hid_;
				gate_diff_t += 4 * num_hid_;
				pre_gate_diff_t += 4 * num_hid_;
			}//for (int n = 0; n < N_; ++n)
			// Backprop output errors to the previous time step
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_, num_hid_, 4 * num_hid_,
				Dtype(1.), pre_gate_diff + pre_gate_.offset(t),
				weight_h, Dtype(0.), h_to_h_.mutable_cpu_data());
			for (int n = 0; n < N_; ++n){
				const bool cont = clip_t ? clip_t[n] : t > 0;
				const Dtype* h_to_h = h_to_h_.cpu_data() + h_to_h_.offset(n);
				if (cont){
					//save the #operations in back-propagation
					//add: combine errors from output_{t} and h_{t+1}
					caffe_add(num_hid_, dh_t_1, h_to_h, dh_t_1);
				}
			}
		}//for (int t = T_ - 1; t >= 0; --t)

		if (this->param_propagate_down_[0]){
			//Gradient w.r.t. input-to-hidden weight
			//dW^{hx} = dgate * x^{t}
			caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4 * num_hid_, input_dim_,
				T_ * N_, Dtype(1.), pre_gate_diff, bottom_data, Dtype(1.),
				this->blobs_[0]->mutable_cpu_diff());
		}
		
		if (this->param_propagate_down_[1]){
			//Gradient w.r.t. hidden-to-hidden weight
			// dW^{hh} = dgate * h^{t-1}
			// here only from 1 to T-1
			caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4 * num_hid_, num_hid_,
				(T_ - 1) * N_, Dtype(1.), pre_gate_diff + pre_gate_.offset(1),
				top_data, Dtype(1.), this->blobs_[1]->mutable_cpu_diff());

			//Add Gradient from previous time-step
			//add gradient generated from h_0
			//why 1 not N_ here?
			caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4 * num_hid_, num_hid_, 1,
				Dtype(1.), pre_gate_diff, h_0_.cpu_data(), Dtype(1.),
				this->blobs_[1]->mutable_cpu_diff());
		}

		if (this->param_propagate_down_[2]){
			// Gradient w.r.t. bias
			// db = average{dg} 
			caffe_cpu_gemv(CblasTrans, T_ * N_, 4 * num_hid_, Dtype(1.),
				pre_gate_diff, bias_multiplier_.cpu_data(), Dtype(1.),
				this->blobs_[2]->mutable_cpu_diff());
		}

		if (propagate_down[0]){
			//Gradient w.r.t. bottom data
			//dX = dg * W^{Xx}
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_ * N_, input_dim_, 4 * num_hid_,
				Dtype(1.), pre_gate_diff, weight_i, Dtype(0.), 
				bottom[0]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(LSTMLayer);
#endif

INSTANTIATE_CLASS(LSTMLayer);
REGISTER_LAYER_CLASS(LSTM);
}