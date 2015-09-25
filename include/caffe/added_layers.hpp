#ifndef CAFFE_ADDED_LAYERS_HPP_
#define CAFFE_ADDED_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/vision_layers.hpp"
namespace caffe{

	template <typename Dtype>
	class LocalLayer : public Layer<Dtype>{
	public:
		explicit LocalLayer(const LayerParameter& param) :Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Local"; }

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const{ return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

		int kernel_size_;
		int stride_;
		int num_;
		int channels_;
		int pad_;
		int height_, width_;
		int height_out_, width_out_;
		int num_output_;
		int bias_term_;

		int M_;
		int K_;
		int N_;

		Blob<Dtype> col_buffer_;
		Blob<Dtype> intermediate_;
		Blob<Dtype> E_;
		Blob<Dtype> weight_diff_temp_;
		Blob<Dtype> x_diff_temp_;
	};

	template <typename Dtype>
	class LocalAdaptiveDropoutLayer : public LocalLayer<Dtype>{
	public:
		explicit LocalAdaptiveDropoutLayer(const LayerParameter& param) : LocalLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LocalAdaptiveDropout"; }

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const{ return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

		//affine parameters of prob weight to hidden weight
		Dtype alpha_;
		Dtype beta_;
		//if we need activate before dropout
		bool need_act_;

		Blob<Dtype> prob_weight_;
		//dropout probability
		Blob<Dtype> prob_vec_;
		//dropout mask after sampling
		Blob<unsigned int> mask_vec_;
		//the raw value of the units before activation
		Blob<Dtype> unact_hidden_;

		LocalAdaptiveDropoutParameter_ActType neuron_act_type_;//act type of hidden units
		LocalAdaptiveDropoutParameter_ActType prob_act_type_;//act type of  probability units

	};

	template <typename Dtype>
	class LocalDropConnectLayer : public LocalLayer<Dtype>{
	public:
		explicit LocalDropConnectLayer(const LayerParameter& param) : LocalLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LocalDropConnect"; }

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const{ return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);

		//if we need activate before dropout
		bool need_act_;


		//dropout parameters
		Blob<Dtype> dropped_weight_;
		Blob<unsigned int> weight_mask_;
		//dropout probability
		Dtype threshold_;

		//scale for training 
		Dtype scale_;
		unsigned int uint_thres_;

	};

}//namespace caffe
#endif
