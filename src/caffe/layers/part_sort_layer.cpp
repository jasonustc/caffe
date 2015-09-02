#include <vector>
#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template<typename Dtype>
	bool greater_cmp(const pair<Dtype, int>& left, const pair<Dtype, int>& right)
	{
		return left.first > right.first;
	}

	template <typename Dtype>
	void PartSortLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const PartSortParameter& part_sort_param = this->layer_param_.part_sort_param();
		part_sort_axis_= bottom[0]->CanonicalAxisIndex(part_sort_param.axis());
		first_element_ = part_sort_param.first_element();
		back_lookup_.resize(bottom[0]->count(0, part_sort_axis_), vector<int>(bottom[0]->count(part_sort_axis_)));
	}

	template <typename Dtype>
	void PartSortLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_LT(part_sort_axis_, bottom[0]->num_axes());
		CHECK_LT(first_element_, bottom[0]->shape(part_sort_axis_));
		top[0]->Reshape(bottom[0]->shape());
	}

	template <typename Dtype>
	void PartSortLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int num = bottom[0]->count(0, part_sort_axis_);
		int dim = bottom[0]->count(part_sort_axis_);
		for (int i = 0; i < num; ++i) {
			std::vector<std::pair<Dtype, int> > bottom_data_vector;
			for (int j = 0; j < dim; ++j) {
				if (j == first_element_)
					continue;
				bottom_data_vector.push_back(
					std::make_pair(bottom_data[i * dim + j], j));
			}
			std::sort(
				bottom_data_vector.begin(), 
				bottom_data_vector.end(),
				greater_cmp<Dtype>);
			top_data[i*dim] = bottom_data[i*dim + first_element_];
			back_lookup_[i][0] = first_element_;
			for (int j = 1; j < dim; j++)
			{
				top_data[i*dim + j] = bottom_data_vector[j-1].first;
				back_lookup_[i][j] = bottom_data_vector[j-1].second;
			}
		}

		/*LOG(INFO) << "PART SORT BOTTOM:";
		for (int i = 0; i < 5; i++)
		{
			LOG(INFO) << bottom_data[i];
		}*/
		/*LOG(INFO) << "TOP";
		for (int i = 0; i < dim; i++)
		{
			LOG(INFO) << top_data[i];
		}*/
	}

	template <typename Dtype>
	void PartSortLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		if (!propagate_down[0])
			return;
		int num = bottom[0]->count(0, part_sort_axis_);
		int dim = bottom[0]->count(part_sort_axis_);
		int bottom_idx = 0;
		for(int i = 0; i < num; ++i) 
		{
			for (int j = 0; j < dim; ++j)
			{
				bottom_idx = back_lookup_[i][j];
				bottom_diff[i*dim + bottom_idx] = top_diff[i*dim + j];
			}
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(PartSortLayer);
#endif

	INSTANTIATE_CLASS(PartSortLayer);
	REGISTER_LAYER_CLASS(PartSort);

}  // namespace caffe
