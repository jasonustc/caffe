#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SSDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  if (this->phase_ == TEST) {
    unrolled_net_->ShareWeightData();
  }
  unrolled_net_->ForwardPrefilled();
}

template <typename Dtype>
void SSDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	//if (!propagate_down[0]) { LOG(INFO) << "NOT BP"; return; }

	unrolled_net_->Backward();
}

INSTANTIATE_LAYER_GPU_FUNCS(SSDLayer);
//INSTANTIATE_LAYER_GPU_FORWARD(SSDLayer);

}  // namespace caffe
