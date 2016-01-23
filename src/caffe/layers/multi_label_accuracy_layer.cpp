#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe{
	template <typename Dtype>
	void MultiLabelAccuracyLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		CHECK(bottom[0]->shape() == bottom[1]->shape()) <<
			"The pred data and labels should have the same shape";
		// Top will contain:
		// top[0] = Sensitivity or Recall (TP / P),
		// top[1] = Specificity (TN / N),
		// top[2] = Harmonic Mean of Sens and Spec, (2/ (P / TP + N / TN)),
		// top[3] = Precision (TP / (TP + FP)),
		// top[4] = F1 Score (2 * TP / (2 * TP + FP + FN))
		top[0]->Reshape(1, 5, 1, 1);
	}

	template <typename Dtype>
	void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Dtype true_positive = 0;
		Dtype false_positive = 0;
		Dtype true_negative = 0;
		Dtype false_negative = 0;
		int count_pos = 0;
		int count_neg = 0;
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_label = bottom[1]->cpu_data();
		const int count = bottom[0]->count();

		for (int i = 0; i < count; i++){
			// Accuracy 
			int label = static_cast<int>(bottom_label[i]);
			if (label > 0){
				//Update Positive accuracy and count
				true_positive += (bottom_data[i] >= 0);
				false_negative += (bottom_data[i] < 0);
				count_pos++;
			}
			if (label < 0){
				//Update Negative accuracy and count
				true_negative += (bottom_data[i] < 0);
				false_positive += (bottom_data[i] >= 0);
				count_neg++;
			}
		}
		Dtype sensitivity = (count_pos > 0) ? (true_positive / count_pos) : 0;
		Dtype specificity = (count_neg > 0) ? (true_negative / count_neg) : 0;
		Dtype harmmean = (count_pos > 0 && count_neg > 0 && true_positive > 0 
			&& true_negative > 0) ?
			2 / (count_pos / true_positive + count_neg / true_negative) : 0;
		Dtype precision = (true_positive > 0) ?
			(true_positive / (true_positive + false_positive)) : 0;
		Dtype f1_score = (true_positive > 0) ?
			2 * true_positive / (2 * true_positive + false_positive + false_negative) : 0;
		DLOG(INFO) << "Sensitivity: " << sensitivity;
		DLOG(INFO) << "Specificity: " << specificity;
		DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
		DLOG(INFO) << "Precision: " << precision;
		DLOG(INFO) << "F1 score: " << f1_score;
		top[0]->mutable_cpu_data()[0] = sensitivity;
		top[0]->mutable_cpu_data()[1] = specificity;
		top[0]->mutable_cpu_data()[2] = harmmean;
		top[0]->mutable_cpu_data()[3] = precision;
		top[0]->mutable_cpu_data()[4] = f1_score;
	}//MultiLabelAccuracy should not be used as a loss function.

	INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
	REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}//namespace caffe
