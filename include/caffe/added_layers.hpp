#ifndef CAFFE_ADDED_LAYERS_HPP_
#define CAFFE_ADDED_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/vision_layers.hpp"
namespace caffe{

	template <typename Dtype>
	class BaseLocalLayer : public Layer<Dtype> {
	public: 
		explicit BaseLocalLayer(const LayerParameter& param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }
	protected:
		void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* output, bool skip_im2col = false);
		void forward_cpu_bias(Dtype* output, const Dtype* bias);
		void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* output);
		void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
			weights);
		void backward_cpu_bias(Dtype* bias, const Dtype* input);
#ifndef CPU_ONLY
		void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
			Dtype* output, bool skip_im2col = false);
		void forward_gpu_bias(Dtype* output, const Dtype* bias);
		void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* col_output);
		void weight_gpu_gemm(const Dtype* col_input, const Dtype* output,
			Dtype* weights);
		void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif
		virtual void compute_output_shape() = 0;

		int kernel_h_, kernel_w_;
		int stride_h_, stride_w_;
		int num_;
		int channels_;
		int pad_h_, pad_w_;
		int height_, width_;
		int group_;
		int num_output_;
		int height_out_, width_out_;
		bool bias_term_;
		bool is_1x1_;
	private:
		inline void local_im2col_cpu(const Dtype* data, Dtype* col_buff){
			im2col_cpu(data, conv_in_channels_, conv_in_height_, conv_in_width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
		}
		inline void local_col2im_cpu(const Dtype* col_buff, Dtype* data){
			col2im_cpu(col_buff, conv_in_channels_, conv_in_height_, conv_in_width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
		}
#ifndef CPU_ONLY
		inline void local_im2col_gpu(const Dtype* data, Dtype* col_buff){
			im2col_gpu(data, conv_in_channels_, conv_in_height_, conv_in_width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
		}
		inline void local_col2im_gpu(const Dtype* col_buff, Dtype* data){
			col2im_gpu(col_buff, conv_in_channels_, conv_in_height_, conv_in_width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
		}
#endif
		int local_out_channels_;
		int local_in_channels_;
		int local_out_spatial_dim_;
		int local_in_height_;
		int local_in_width_;
		int kernel_dim_;
		int weight_offset_;
		int col_offset_;
		int output_offset_;

		Blob<Dtype> col_buffer_;
		Blob<Dtype> bias_multiplier_;
	};

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
