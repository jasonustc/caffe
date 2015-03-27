#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void VideoUnrollLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//top[0]->Reshape(this->layer_param_.video_unroll_param().shape());
	int frame_channels = this->layer_param_.video_unroll_param().frame_channels();
	CHECK(bottom[0]->shape(1) % frame_channels == 0) << "all channels of a video cannot be divisible by frame_channels.";
	int frames_per_video = bottom[0]->shape(1) / frame_channels;
	top[0]->Reshape(bottom[0]->shape(0)*frames_per_video, frame_channels,
		bottom[0]->shape(2), bottom[0]->shape(3));
	//continue indicators, only one stream
	top[1]->Reshape(top[0]->shape(0), 1, 1, 1);
	//frames per video
	top[1]->mutable_cpu_data()[0] = frames_per_video;

	top[2]->Reshape(1, 1, 1, 1);
	Dtype* top_data = top[0]->mutable_cpu_data();
	for (int i = 0; i < top[1]->count(); i++)
	{
		if (i%frames_per_video == 0)
			top_data[i] = 0;
		else
			top_data[i] = 1;
	}

	CHECK_EQ(top[0]->count(), bottom[0]->count())
		<< "new shape must have the same count as input";
	top[0]->ShareData(*bottom[0]);
	top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(VideoUnrollLayer);
REGISTER_LAYER_CLASS(VideoUnroll);

}  // namespace caffe
