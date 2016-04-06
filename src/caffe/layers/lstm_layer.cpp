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

			//input-to-hidden weights
			//initialize the weights
			//TODO: compute the four weights indepently
			vector<int> weight_shape;
			weight_shape.push_back(4 * num_hid_);
			weight_shape.push_back(input_dim_);
		}
	}
}