#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void VideoLabelExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	int frames_per_video = (int)bottom[1]->cpu_data()[0];
	vector<int> shape;
	shape.push_back(bottom[0]->count()*frames_per_video);
	top[0]->Reshape(shape);
}

template <typename Dtype>
void VideoLabelExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	int frames_per_video = (int)bottom[1]->cpu_data()[0];
	Dtype* top_data = top[0]->mutable_cpu_data();

	for (int i = 0; i < bottom[0]->count(); i++)
	{
		caffe_set(frames_per_video, (Dtype)bottom_data[i], &(top_data[i*frames_per_video]));
	}

}

INSTANTIATE_CLASS(VideoLabelExpandLayer);
REGISTER_LAYER_CLASS(VideoLabelExpand);

}  // namespace caffe
