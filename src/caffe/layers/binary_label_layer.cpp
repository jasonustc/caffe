#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void BinaryLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	caffe_copy(bottom[0]->count(), bottom_data, top_data);

	int label = 0;

	for (int i = 0; i < bottom[0]->count(); i++)
	{
		top_data[i] = (top_data[i] == 0 ? 0 : 1);
	}
}

INSTANTIATE_CLASS(BinaryLabelLayer);
REGISTER_LAYER_CLASS(BinaryLabel);

}  // namespace caffe
