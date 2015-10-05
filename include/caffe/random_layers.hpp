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
#include "caffe/neuron_layers.hpp"

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

	/*
	 * since we already have exp layer and innerproduct layer in caffe
	 * we directly use them to calculate u_t = W * input + b and \sigma_t = exp(W * input + b)
	 * so here we take vector u and vector sigma as input, and use them to sample gaussian values
	 */
	template <typename Dtype>
	class SamplingLayer : public Layer<Dtype> {
	public:
		explicit SamplingLayer(const LayerParameter& param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		inline virtual const char* type(){ return "SamplingLayer"; }

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

		virtual inline int ExactNumBottomBlobs(){ return 2; }
		virtual inline int ExactNumTopBlobs(){ return 1; }

		//the sampling value of standard gaussian distribution
		//needed in backpropagation
		Blob<Dtype> gaussian_value_;
	};
}

#endif