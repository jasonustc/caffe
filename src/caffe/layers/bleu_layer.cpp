#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void BLEULayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	}

	INSTANTIATE_CLASS(AccuracyLayer);
	REGISTER_LAYER_CLASS(Accuracy);
}//namespace caffe