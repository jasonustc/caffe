/*
** Created by qing li, in this file,
** there are all the declarations of layers used in video analysis
** maybe more detailed documents are needed
*/

#ifndef CAFFE_VIDEO_LAYERS_HPP_
#define CAFFE_VIDEO_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/**
	* @brief Takes one blob, pool it on certain axis
	*/
	template <typename Dtype>
	class AxisPoolingLayer : public Layer<Dtype> {
	public:
		explicit AxisPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AxisPooling"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		* @param top output Blob vector (length 1)
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);*/

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		/*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/

		int pool_axis_;
		int num_pools_;
		int pool_input_size_;
		Blob<int> max_idx_;
	};

	/**
	* @brief ConstrainIP with constraints, used in SSD layer
	* Reference:
	*   [1] Hoai, Minh, and Andrew Zisserman. "Improving human action recognition using score distribution and ranking."
	*       Proceedings of the Asian Conference on Computer Vision. 2014.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class ConstrainIPLayer : public Layer<Dtype> {
	public:
		explicit ConstrainIPLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ConstrainIP"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int M_;
		int K_;
		int N_;
		bool bias_term_;
		Blob<Dtype> bias_multiplier_;
		Dtype sum1_rate_;
		Dtype monotonic_rate_;
	};

	/**
	* @brief Added by qing li, Like reshape_layer, roll frames to video, and generate continuing indicators used by SSD layers
	*/
	template <typename Dtype>
	class FramesRollLayer : public Layer<Dtype> {
	public:
		explicit FramesRollLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "FramesRoll"; }
		virtual inline int MinNumBottomBlobs() const { return 1; }
		virtual inline int MaxNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){}
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){}
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>&  top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

	};
	
	/**
	* @brief Takes one blob, sort it on certain axis
	*/
	template <typename Dtype>
	class SortLayer: public Layer<Dtype> {
	public:
		explicit SortLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Sort"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		* @param top output Blob vector (length 1)
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);*/

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		/*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/

		int sort_axis_;
		// back_lookup_[n][i] = the index in the bottom blob of the i_th element of the n_th sample in top blob 
		vector<vector<int>> back_lookup_;

	/*private:
		static inline bool greater_cmp(const pair<Dtype, int>& left, const pair<Dtype, int>& right)
		{
			return left.first > right.first;
		}*/
	};

	/**
	 * @brief SSD layer = Slice + Sort + ConstrainIP
	 *
	 * References
	 * [1] Hoai, Minh, and Andrew Zisserman. "Improving human action recognition using score distribution and ranking."
	 *     Proceedings of the Asian Conference on Computer Vision. 2014.
	 */
	template <typename Dtype>
	class SSDLayer : public Layer<Dtype>
	{
	public:
		explicit SSDLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SSD"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		 * @brief Fills net_param with the SSD network arcthiecture.
		 */
		virtual void FillUnrolledNet(NetParameter* net_param) const;

		/**
		 * TODO: explain the forward pass more carefully
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief A helper function, useful for stringifying category indices.
		virtual string int_to_str(const int t) const;

		/// @brief A Net to implement the SSD functionality.
		shared_ptr<Net<Dtype> > unrolled_net_;


		/**
		* @brief The number of categorys in the layer's input 
		*/
		int C_;
		int cate_axis_;
		Dtype constrain_ip_sum1_rate_;
		Dtype constrain_ip_monotonic_rate_;

		Blob<Dtype>* x_input_blob_;
		Blob<Dtype>* x_output_blob_;

	};


	/**
	* @brief Takes one blob, partially sort it on certain axis
	*/
	template <typename Dtype>
	class PartSortLayer: public Layer<Dtype> {
	public:
		explicit PartSortLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "PartSort"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		* @param top output Blob vector (length 1)
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);*/

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		/*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/

		int part_sort_axis_;
		int first_element_;
		// back_lookup_[n][i] = the index in the bottom blob of the i_th element of the n_th sample in top blob 
		vector<vector<int>> back_lookup_;
	};

	/**
	* @brief SSD layer = Slice + PartSort + ConstrainIP
	*
	* References
	* [1] Hoai, Minh, and Andrew Zisserman. "Improving human action recognition using score distribution and ranking."
	*     Proceedings of the Asian Conference on Computer Vision. 2014.
	*/
	template <typename Dtype>
	class RCSLayer : public Layer<Dtype>
	{
	public:
		explicit RCSLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "RCS"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		/**
		* @brief Fills net_param with the RCS network arcthiecture.
		*/
		virtual void FillUnrolledNet(NetParameter* net_param) const;

		/**
		* TODO: explain the forward pass more carefully
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief A helper function, useful for stringifying category indices.
		virtual string int_to_str(const int t) const;

		/// @brief A Net to implement the RCS functionality.
		shared_ptr<Net<Dtype> > unrolled_net_;


		/**
		* @brief The number of categorys in the layer's input
		*/
		int C_;
		int cate_axis_;

		Blob<Dtype>* x_input_blob_;
		Blob<Dtype>* x_output_blob_;

	};


	/**
	* @brief Added by qing li, Like reshape_layer, unroll video to frames, and generate continuing indicators used by RCS layers
	*/
	template <typename Dtype>
	class VideoUnrollLayer : public Layer<Dtype> {
	public:
		explicit VideoUnrollLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoUnroll"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MinNumTopBlobs() const { return 2; }
		virtual inline int MaxBottomBlobs() const { return 3; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){}
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){}
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>&  top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

	};


	/**
	* @brief Added by qing li, Like reshape_layer, unroll video to frames, and generate continuing indicators used by RCS layers
	*/
	template <typename Dtype>
	class VideoLabelExpandLayer : public Layer<Dtype> {
	public:
		explicit VideoLabelExpandLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoLabelExpand"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

	};

	/**
	* @brief Added by qing li
	*/
	template <typename Dtype>
	class VideoSigmoidLabelLayer : public Layer<Dtype> {
	public:
		explicit VideoSigmoidLabelLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoSigmoidLabel"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

		int n_cate_;

	};


	/**
	* @brief Added by qing li
	*/
	template <typename Dtype>
	class BinaryLabelLayer : public Layer<Dtype> {
	public:
		explicit BinaryLabelLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BinaryLabel"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

	};

	///**
	//* @brief normalize the sample to [0,1]
	//*/
	//template <typename Dtype>
	//class Norm01Layer : public NeuronLayer<Dtype> {
	//public:
	//	explicit Norm01Layer(const LayerParameter& param)
	//		: NeuronLayer<Dtype>(param) {}
	//	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	//		const vector<Blob<Dtype>*>& top);

	//	virtual inline const char* type() const { return "Norm01"; }

	//protected:
	//	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	//		const vector<Blob<Dtype>*>& top);
	//	
	//	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	//		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	//
	//};
}  // namespace caffe

#endif  // CAFFE_VIDEO_LAYERS_HPP_
