#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RecurrentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// Hacky fix for test time... reshare all the shared blobs.
	// TODO: somehow make this work non-hackily.
	//share weight data and diff with its owner
	if (this->phase_ == TEST) {
		unrolled_net_->ShareWeightData();
	}
//	LOG(ERROR) << "reshape_l1_norm: " << bottom[0]->asum_data();
//	LOG(ERROR) << "reshape_l2_norm: " << bottom[0]->sumsq_data();

//	if (top.size() > 1){
//		LOG(ERROR) << "h_T_l1_norm: " << top[1]->asum_data();
//		LOG(ERROR) << "h_T_l2_norm: " << top[1]->sumsq_data();
//		LOG(ERROR) << "c_T_l1_norm: " << top[2]->asum_data();
//		LOG(ERROR) << "c_T_l2_norm: " << top[2]->sumsq_data();
//		LOG(ERROR) << "W_xc_l1_norm: " << this->blobs_[0]->asum_data();
//		LOG(ERROR) << "W_xc_l2_norm: " << this->blobs_[0]->sumsq_data();
//		LOG(ERROR) << "bc_l1_norm: " << this->blobs_[1]->asum_data();
//		LOG(ERROR) << "bc_l2_norm: " << this->blobs_[1]->sumsq_data();
//		LOG(ERROR) << "W_hc_l1_norm: " << this->blobs_[2]->asum_data();
//		LOG(ERROR) << "W_hc_l2_norm: " << this->blobs_[2]->sumsq_data();
//	}

	DCHECK_EQ(recur_input_blobs_.size(), recur_output_blobs_.size());
	for (int i = 0; i < recur_input_blobs_.size(); ++i) {
		const int count = recur_input_blobs_[i]->count();
		DCHECK_EQ(count, recur_output_blobs_[i]->count());
		//during training: copy c_T, h_T to c_0, h_0
		//to make sure that if the end of current batch is not 
		//an end of the sequence
		const Dtype* timestep_T_data = recur_output_blobs_[i]->gpu_data();
		Dtype* timestep_0_data = recur_input_blobs_[i]->mutable_gpu_data();
		caffe_copy(count, timestep_T_data, timestep_0_data);
	}

	unrolled_net_->ForwardPrefilled();
}

INSTANTIATE_LAYER_GPU_FORWARD(RecurrentLayer);

}  // namespace caffe
