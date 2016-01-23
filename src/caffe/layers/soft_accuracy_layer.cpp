#include <functional>
#include <utility>
#include <vector>

#include "caffe/loss_layers.hpp"

namespace caffe{
	template<typename Dtype>
	void SoftAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		err_gap_ = this->layer_param_.accuracy_param().err_gap();
		CHECK_GE(err_gap_, 0) << "error gap must be non-negative";
	}

	template<typename Dtype>
	void SoftAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->count(1), 1) << "predicted score should have only one dimension";
		CHECK_EQ(bottom[1]->count(1), 1) << "groundtruth score should have only one dimension";
		CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "predict score and groundtruth score should have"
			<< " the same number";
		//accuracy is a scalar; 0 axes
		vector<int> top_shape(0);
		top[0]->Reshape(top_shape);
		if (top.size() > 1){
			//output square error to top[1]
			top[1]->Reshape(top_shape);
		}
	}

	template<typename Dtype>
	void SoftAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num = bottom[0]->num();
		const Dtype* pred_data = bottom[0]->cpu_data();
		const Dtype* true_data = bottom[1]->cpu_data();
		Dtype accuracy = 0;
		Dtype sq_err = 0;
		bool compute_sq_error = top.size() > 1;
		Dtype abs_err;
		for (int i = 0; i < num; i++){
			abs_err = abs(pred_data[i] - true_data[i]);
			if (abs_err < err_gap_){
				++accuracy;
			}
			if (compute_sq_error){
				sq_err += abs_err * abs_err;
			}
		}
		top[0]->mutable_cpu_data()[0] = accuracy / num;
		if (compute_sq_error){
			top[1]->mutable_cpu_data()[0] = sq_err / num;
		}
	}
	//SoftAccuracy Layer should not be used as a loss function
	INSTANTIATE_CLASS(SoftAccuracyLayer);
	REGISTER_LAYER_CLASS(SoftAccuracy);
}