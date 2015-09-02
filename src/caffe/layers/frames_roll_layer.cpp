#include <vector>

#include "caffe/blob.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void FramesRollLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(!((bottom.size() == 2) && (this->layer_param_.frames_roll_param().frames_per_video() > 0))) 
		<< "Cannot simultaneously input frames per video and set the frames per video param ";
	int frames_per_video = this->layer_param_.frames_roll_param().frames_per_video();
	if (bottom.size() == 2)
	{
		frames_per_video = bottom[1]->cpu_data()[0];
	}

	CHECK((bottom[0]->shape(0) % frames_per_video) == 0)
		<< "sum of frames is not divisible by frames_per_video";
	
	int video_num = bottom[0]->shape(0) / frames_per_video;
	// [video_num, frames, streams, feature]
	top[0]->Reshape(video_num, frames_per_video, bottom[0]->shape(1), bottom[0]->count(2));
	CHECK_EQ(top[0]->count(), bottom[0]->count())
		<< "new shape must have the same count as input";
	top[0]->ShareData(*bottom[0]);
	top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(FramesRollLayer);
REGISTER_LAYER_CLASS(FramesRoll);

}