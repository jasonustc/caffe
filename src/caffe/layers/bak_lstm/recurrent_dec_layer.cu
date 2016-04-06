/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/10/4
** desc: DecodingRecurrentLayer(GPU)
*********************************************************************************/
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
	LOG(ERROR) << "bottom gradient norm1 " << bottom[0]->asum_diff();
	LOG(ERROR) << "bottom data norm1 " << bottom[0]->asum_data();
	LOG(ERROR) << "W_hc gradient norm1 " << this->blobs_[0]->asum_diff();
	LOG(ERROR) << "W_hc data norm1 " << this->blobs_[0]->asum_data();
	LOG(ERROR) << "b_c gradient norm1 " << this->blobs_[1]->asum_diff();
	LOG(ERROR) << "b_c data norm1 " << this->blobs_[1]->asum_data();
	LOG(ERROR) << "W_xc gradient norm1 " << this->blobs_[2]->asum_diff();
	LOG(ERROR) << "W_xc data norm1 " << this->blobs_[2]->asum_data();
}

INSTANTIATE_LAYER_GPU_FORWARD(DRecurrentLayer);

}  // namespace caffe
