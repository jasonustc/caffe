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
#include "caffe/util/transform.hpp"
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

	template <typename Dtype>
	class RandomTransformLayer : public Layer<Dtype>{
	public: 
		explicit RandomTransformLayer(const LayerParameter& param) : Layer<Dtype>(param){}
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

		//added by xu shen here
		//get the coordination matrix after transformation
		void GetTransCoord();

		bool rotation_;
		bool scale_;
		bool shift_;

		int Height_;
		int Width_;

		float start_angle_;
		float end_angle_;
		float start_scale_;
		float end_scale_;
		float dx_prop_;
		float dy_prop_;

		float curr_scale_;
		float curr_angle_;
		//shift by #pixels
		float curr_shift_x_;
		float curr_shift_y_;

		Border BORDER_; // border type
		Interp INTERP_; //interpolation type
		//3x3 transform matrix buffer, row order
		float tmat_[9];

		//Indices for image transformation
		//We use blob's data to be fwd and diff to be backward indices
		shared_ptr<Blob<float>> coord_idx_;

	};

	template <typename Dtype>
	class SamplingLayer : public NeuronLayer<Dtype> {
	public:
		explicit SamplingLayer(const LayerParameter& param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
//		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type(){ return "SamplingLayer"; }

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

		virtual inline bool EqualNumBottomTopBlobs{ return true; }

		//sampling parameters of gaussian distribution
		float mu_;
		float sigma_;
	};
}

#endif