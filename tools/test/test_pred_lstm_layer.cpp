#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/loss_layers.hpp"

namespace caffe{
	template <typename Dtype>
	class PredLSTMTest{
	public:
	protected:
		Blob<Dtype>* c_0_;
		Blob<Dtype>* h_0_;
		Blob<Dtype>* cont_0_;

		Blob<Dtype>* c_T_;
		Blob<Dtype>* h_T_;
		Blob<Dtype>* h_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;
	};
}