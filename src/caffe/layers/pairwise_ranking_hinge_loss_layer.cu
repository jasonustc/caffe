#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
using std::max;

namespace caffe {

template <typename Dtype>
void PairwiseRankingHingeLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  return Forward_cpu(bottom, top);
}

template <typename Dtype>
void PairwiseRankingHingeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // TODO(Yangqing): implement the GPU version of softmax.
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseRankingHingeLossLayer);

}  // namespace caffe
