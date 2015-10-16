#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DRecurrentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
	if (this->phase_ == TEST) {
		unrolled_net_->ShareWeightData();
	}

	unrolled_net_->ForwardPrefilled();
}

INSTANTIATE_LAYER_GPU_FORWARD(DRecurrentLayer);

}  // namespace caffe
