// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//reshape the top layer to shape like bottom layer
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  drop_type_ = this->layer_param_.dropout_param().drop_type();
  a_ = this->layer_param_.dropout_param().a();
  b_ = this->layer_param_.dropout_param().b();
  mu_ = this->layer_param_.dropout_param().mu();
  sigma_ = this->layer_param_.dropout_param().sigma();
  //debug only check, not check in non-debug mode
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  DCHECK(a_ < b_);
  DCHECK(sigma_ > 0.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
	  if (this->drop_type_ == DropoutParameter_DROPTYPE_UNIFORM){
		caffe_rng_uniform(count, (Dtype)a_, (Dtype)b_, mask);
		scale_ = 2. / (b_ - a_);
	  }
	  else if(this->drop_type_ == DropoutParameter_DROPTYPE_GAUSSIAN){
		caffe_rng_gaussian(count, (Dtype)mu_, (Dtype)sigma_, mask);
		scale_ = 1. / mu_;
	  }
	  else{
		caffe_rng_bernoulli_c<Dtype>(count, 1. - threshold_, mask);
	  }
	  for (int i = 0; i < count; ++i) {
		  top_data[i] = bottom_data[i] * mask[i] * scale_;
	  }
	  rand_vec_.PrintDataToFile("mask_of_dropout");
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (this->phase_ == TRAIN) {
			const Dtype* mask = rand_vec_.cpu_data();
			const int count = bottom[0]->count();
			for (int i = 0; i < count; ++i) {
				bottom_diff[i] = top_diff[i] * mask[i] * scale_;
			}
			top[0]->PrintDiffToFile("top_dropout");
			bottom[0]->PrintDiffToFile("bottom_dropout");
		}
		else {
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
