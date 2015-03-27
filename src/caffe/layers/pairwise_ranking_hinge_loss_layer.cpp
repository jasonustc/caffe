#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"


namespace caffe {
	template <typename Dtype>
	void PairwiseRankingHingeLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			LossLayer<Dtype>::LayerSetUp(bottom, top);

			CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
			CHECK_EQ(bottom[0]->height(), 1);
			CHECK_EQ(bottom[0]->width(), 1);
			CHECK_EQ(bottom[1]->height(), 1);
			CHECK_EQ(bottom[1]->width(), 1);
			diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
			diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
			dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
			// vector of ones used to sum along channels
			summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
			for (int i = 0; i < bottom[0]->channels(); ++i)
				summer_vec_.mutable_cpu_data()[i] = Dtype(1);
	}

	template <typename Dtype>
	void PairwiseRankingHingeLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			//const Dtype* query_data = bottom[0]->cpu_data();
			//const Dtype* similar_sample_data = bottom[1]->cpu_data();
			//Dtype* similar_sample_diff = bottom[1]->mutable_cpu_diff();

			////LOG(INFO)<<"SCORE:"<<*query_data<<":"<<*similar_sample_data; //added by qing li
			//
			//int num = bottom[0]->num();
			//int count = bottom[0]->count();
			//int dim = count / num;
			//caffe_sub(count, query_data, similar_sample_data,similar_sample_diff);
			//Dtype margin = this->layer_param_.pairwise_ranking_hinge_loss_param().margin();

			//int r_count=0;
			//for(int i=0;i<num;i++)
			//{
			//	if(query_data[i]-similar_sample_data[i]>=margin)
			//		r_count++;
			//	
			//}
			//LOG(INFO)<<"r_count:"<<r_count;
			//LOG(INFO)<<"NUM:"<<num;
			//LOG(INFO)<<"accuracy:"<<(float)r_count/num;  //added by qing li

			//Dtype loss = 0;
			//Dtype query_similar_distance_norm;
			//for (int i = 0; i < num; ++i) {
			//	/* query_similar_distance_norm = caffe_cpu_dot(
			//	dim, similar_sample_diff + bottom[1]->offset(i),
			//	similar_sample_diff + bottom[1]->offset(i));*/
			//	Dtype* temp_dis=similar_sample_diff + bottom[1]->offset(i);
			//	/*query_similar_distance_norm = temp_dis[0]*temp_dis[0];*/
			//	query_similar_distance_norm = temp_dis[0];
			//	Dtype temp_max=max(Dtype(0), -query_similar_distance_norm 
			//		+ margin);
			//	loss +=temp_max*temp_max ;
			//	LOG(INFO)<<"query_similar_distance_norm:"<<query_similar_distance_norm;
			//	LOG(INFO)<<"margin:"<<margin;
			//	LOG(INFO)<<"temp_max:"<<temp_max;  //added by qing li
			//}
			//(*top)[0]->mutable_cpu_data()[0] = loss;
			//return loss / num;
			int count = bottom[0]->count();
			caffe_sub(
				count,
				bottom[0]->cpu_data(),  // a
				bottom[1]->cpu_data(),  // b
				diff_.mutable_cpu_data());  // a_i-b_i
			//LOG(INFO)<<"a:"<<bottom[0]->cpu_data()[0];
			//LOG(INFO)<<"b:"<<bottom[1]->cpu_data()[0]; //added by qing li

			const int channels = bottom[0]->channels();
			Dtype margin = this->layer_param_.pairwise_ranking_hinge_loss_param().margin();
			Dtype loss(0.0);
			for (int i = 0; i < bottom[0]->num(); ++i) {
				dist_sq_.mutable_cpu_data()[i] = diff_.cpu_data()[i];
				Dtype tmp(0.0);
				loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
				
				//LOG(INFO)<<i<<":"<<dist_sq_.cpu_data()[i]; //added by qing li
			}
			loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
			top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void PairwiseRankingHingeLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			Dtype margin = this->layer_param_.pairwise_ranking_hinge_loss_param().margin();
			for (int i = 0; i < 2; ++i) {
				if (propagate_down[i]) {
					const Dtype sign = (i == 0) ? 1 : -1;
					const Dtype alpha = sign * Dtype(1.0) /
						static_cast<Dtype>(bottom[i]->num());
					//LOG(INFO)<<"alpha:"<<alpha;
					int num = bottom[i]->num();
					int channels = bottom[i]->channels();
					for (int j = 0; j < num; ++j) {
						Dtype* bout = bottom[i]->mutable_cpu_diff();
						//LOG(INFO)<<"dist:"<<dist_sq_.cpu_data()[j];//added by qing li
						if ((margin-dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
							caffe_set(channels, -alpha, bout+(j*channels));
						} else {

							caffe_set(channels, Dtype(0), bout + (j*channels));
						}
					}
				}
			}
	}



#ifdef CPU_ONLY
	STUB_GPU(PairwiseRankingHingeLossLayer);
#endif

	INSTANTIATE_CLASS(PairwiseRankingHingeLossLayer);
	REGISTER_LAYER_CLASS(PairwiseRankingHingeLoss);

}  // namespace caffe