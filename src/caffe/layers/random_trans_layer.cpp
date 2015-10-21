#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/random_layers.hpp"
//#include "caffe/util/imshow.hpp"

//#include <opencv2/highgui/highgui.hpp>

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Layer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom.size(), 1) << "RandomTranform Layer only takes one single blob as input.";
		CHECK_EQ(top.size(), 1) << "RandomTransform Layer only takes one single blob as output.";

		LOG(INFO) << "Random Transform Layer using border type: " << this->layer_param_.rand_trans_param().border()
			<< ", using interpolation: " << this->layer_param_.rand_trans_param().interp();
		rotation_ = this->layer_param_.rand_trans_param().has_start_angle() &&
			this->layer_param_.rand_trans_param().has_end_angle();
		shift_ = this->layer_param_.rand_trans_param().has_dx_prop() &&
			this->layer_param_.rand_trans_param().has_dy_prop();
		scale_ = this->layer_param_.rand_trans_param().has_start_scale()&&
			this->layer_param_.rand_trans_param().has_end_scale();
		
		if (rotation_){
			start_angle_ = this->layer_param_.rand_trans_param().start_angle();
			end_angle_ = this->layer_param_.rand_trans_param().end_angle();
			CHECK_GE(start_angle_, 0);
			CHECK_LE(end_angle_, 360);
			CHECK_LE(start_angle_, end_angle_);
			LOG(INFO) << "random rotate in [" << start_angle_ << "," << end_angle_ << "].";
		}
		if (shift_){
			dx_prop_ = this->layer_param_.rand_trans_param().dx_prop();
			dy_prop_ = this->layer_param_.rand_trans_param().dy_prop();
			CHECK_GE(dx_prop_, 0);
			CHECK_LE(dx_prop_, 1);
			CHECK_GE(dy_prop_, 0);
			CHECK_LE(dy_prop_, 1);
			LOG(INFO) << "Random shift image by dx <= " << dx_prop_ << "\%, dy <= " << dy_prop_ << "\%.";
		}
		if (scale_){
			start_scale_ = this->layer_param_.rand_trans_param().start_scale();
			end_scale_ = this->layer_param_.rand_trans_param().end_scale();
			CHECK_GT(start_scale_, 0);
			CHECK_LE(start_scale_, end_scale_);
			LOG(INFO) << "Random scale image in [" << start_scale_ << "," << end_scale_ << "]";
		}

		Height_ = bottom[0]->height();
		Width_ = bottom[0]->width();
		BORDER_ = static_cast<Border>(this->layer_param_.rand_trans_param().border());
		INTERP_ = static_cast<Interp>(this->layer_param_.rand_trans_param().interp());
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//TODO: shape the parameter blobs
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::GetTransCoord(){
		//compute transformation matrix
		if (rotation_){
			//randomly generate rotation angle
			caffe_rng_uniform(1, start_angle_, end_angle_, &curr_angle_);
			TMatFromParam(0, curr_angle_, curr_angle_, tmat_);
		}
		if (scale_){
			caffe_rng_uniform(1, start_scale_, end_scale_, &curr_scale_);
			curr_scale_ = Dtype(1.);
			TMatFromParam(1, curr_scale_, curr_scale_, tmat_);
		}
		if (shift_){
			float shift_pixels_x = dx_prop_ * Width_;
			float shift_pixels_y = dy_prop_ * Height_;
			caffe_rng_uniform(1, -shift_pixels_x, shift_pixels_x, &curr_shift_x_);
			caffe_rng_uniform(1, -shift_pixels_y, shift_pixels_y, &curr_shift_y_);
			TMatFromParam(2, curr_shift_x_, curr_shift_y_, tmat_);
		}
		//Canoincal size is set, so after finding the transformation,
		//crop or pad to that canonical size.
		//First find the coordinate matrix for this transformation
		//do not have diff to save memory
		int height_new(0), width_new(0);
		Blob<float> *original_coord = new Blob<float>();
		GenCoordMat(tmat_, Height_, Width_, original_coord, height_new, width_new, BORDER_, INTERP_);
//		LOG(INFO) << "Cropping " << height_new << " by " << width_new
//			<< " to input size: "
//			<< Height_ << "x" << Width_;
		//Crop the coordinates at the center to this size.
		const int orig_width = width_new;
		const int orig_height = height_new;
		//Need to set coordinates to the size of crop_size.
		//here allocate memory every time is very expensive
		switch (INTERP_){
		case NN:
			coord_idx_.reset(new Blob<float>(1, 1, Height_ * Width_, 1));
			break;
		case BILINEAR:
			coord_idx_.reset(new Blob<float>(1, 1, Height_ * Width_ * 4, 1));
			break;
		default:
			LOG(FATAL) << "Unkown pooling method.";
		}
		ImageSize original(orig_width, orig_height);
		ImageSize target(Width_, Height_);
		CropCenter(original_coord->cpu_data(), original, target, INTERP_,
			coord_idx_->mutable_cpu_data());
		//clear up.
		delete original_coord;
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int channels = bottom[0]->channels();
		//if there are no random transformations, we just copy bottom data to top blob
		//in test phase, we just don't do any transformations
		if ((!shift_ && !scale_ && !rotation_) || this->phase_ == TEST){
			caffe_copy(count, bottom_data, top_data);
		}
		else{
			GetTransCoord();
			//apply Interpolation on bottom_data using tmat_[i] into top_data
			//the coord_idx_[i] will be of size as the output data
			InterpImageNN_cpu(bottom[0], coord_idx_->cpu_data(), top[0], INTERP_);
		}
	}

	//Backward has to return 1 bottom
	//Note that backwards coordinate indices are also stored in data
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		//Reset bottom diff.
		const Dtype* top_diff = top[0]->cpu_diff();
		const int count = top[0]->count();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(count, Dtype(0.), bottom_diff);
		if ((!shift_ && !rotation_ && !scale_)){
			caffe_copy(count, top_diff, bottom_diff);
		}
		else{
			PropagateErrorNN_cpu(top[0], coord_idx_->cpu_data(), bottom[0], INTERP_);
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(RandomTransformLayer);
#endif

	INSTANTIATE_CLASS_FLOAT_ONLY(RandomTransformLayer);

	//since the atomicAdd gpu function in transform only support float,
	//so we only register float functions here
	REGISTER_LAYER_CLASS_FLOAT_ONLY(RandomTransform);
}