#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void VideoLabelExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	int frames_per_video = (int)bottom[1]->cpu_data()[0];
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count(1);
	vector<int> shape;
	shape.push_back(num * frames_per_video);
	shape.push_back(dim);
	top[0]->Reshape(shape);
}

template <typename Dtype>
void VideoLabelExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	int frames_per_video = (int)bottom[1]->cpu_data()[0];
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int num = bottom[0]->num();
	//to allow multiple labels for one single sample
	const int dim = bottom[0]->count(1);

	for (int i = 0; i < bottom[0]->num(); i++)
	{
		for (int j = 0; j < frames_per_video; j++){
			caffe_copy(dim, bottom_data + bottom[0]->offset(i), 
				top_data + top[0]->offset(i * frames_per_video + j));
		}
	}
}

INSTANTIATE_CLASS(VideoLabelExpandLayer);
REGISTER_LAYER_CLASS(VideoLabelExpand);

}  // namespace caffe
