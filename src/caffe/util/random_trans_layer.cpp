#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/imshow.hpp"

//#include <opencv2/highgui/highgui.hpp>

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom.size(), 1) << "RandomTranform Layer only takes one single blob as input.";
		CHECK_EQ(top.size(), 1) << "RandomTransform Layer only takes one single blob as output.";

		LOG(INFO) << "Random Transform Layer usring " << this->layer_param_.name()
			<< "using interpolation: " << this->layer_param_.transformations(0).interp();
		scale_ = this->layer_param_.rand_trans_param().scale();
		shift_ = this->layer_param_.rand_trans_param().shift();
		rotation = this->layer_param_.rand_trans_param().rotation();

		start_angle_ = this->layer_param_.rand_trans_param().start_angle();
		end_angle_ = this->layer_param_.rand_trans_param().end_angle();
		scale_coeff_ = this->layer_param_.rand_trans_param().scale_coeff();
		dx_prop_ = this->layer_param_.rand_trans_param().dx_prop();
		dy_prop_ = this->layer_param_.rand_trans_param().dy_prop();
	}
}