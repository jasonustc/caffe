#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void VideoSigmoidLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	n_cate_ = 101;
	vector<int> shape;
	shape.push_back(bottom[0]->shape(0));
	shape.push_back(n_cate_);
	top[0]->Reshape(shape);
}

template <typename Dtype>
void VideoSigmoidLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	caffe_set(top[0]->count(), Dtype(0), top_data);

	for (int i = 0; i < bottom[0]->count(); i++)
	{
		int label = bottom_data[i];
		top_data[i*n_cate_ + label] = Dtype(1);
	}
}

INSTANTIATE_CLASS(VideoSigmoidLabelLayer);
REGISTER_LAYER_CLASS(VideoSigmoidLabel);

}  // namespace caffe
