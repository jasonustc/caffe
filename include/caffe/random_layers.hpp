#ifndef CAFFE_RANDOM_LAYERS_HPP_
#define CAFFE_RANDOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	template <typename Dtype> class NoiseLayer;

	/**
	 * @brief To randomly add noise to the input
	 * In auto-encoder, this can train a generative model
	 * to learn the distribution of the input rather than
	 * just the code.
	 */
	template <typename Dtype>
	class NoiseLayer : public Layer<Dtype> {
	public:
		/*
		 *@param distribution type and corresponding parameters
		 */
		explicit NoiseLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Noise"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		virtual inline int ExtactNumBottomBlobs() const { return 1; }
		virtual inline int ExtactNumTopBlobs() const { return 1; }

		//parameter for noise distribution
		Dtype alpha_;
		Dtype beta_;

		//noise type
		NoiseParameter_NoiseType noise_type_;

		//apply type
		NoiseParameter_ApplyType apply_type_;

		//noise value
		Blob<Dtype> noise_;
	};

	template <typename Dtype>
	class RandomTransformLayer : public Layer<Dtype>{
	public: 
		RandomTransformLayer(const LayerParameter& param) :Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type(){ return "random_transform"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& top);


		virtual inline bool EqualNumBottomTopBlobs() const { return true; }
		virtual inline int ExtactNumBottomBlobs() const { return 1; }
		virtual inline int ExtactNumTopBlobs() const { return 1; }

		Blob<Dtype> trans_mat; //3 x 3 transform mat
		Blob<Dtype> trans_coord; //the corresponding coordinate matrix index for transform

		bool rotation_;
		bool scale_;
		bool shift_;

		float start_angle_;
		float end_angle_;
		float scale_coeff_;
		float dx_prop_;
		float dy_prop_;

	};
}

#endif