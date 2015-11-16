#ifndef DMODEL_LOADER_H_
#define DMODEL_LOADER_H_

#include <stdio.h>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/net.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;

using boost::shared_ptr;
using std::string;

template <typename Dtype>
class DModelLoader
{
public:
	DModelLoader();
	~DModelLoader();

	bool LoadModel(
		string vgg_mean_path,   // vgg_mean.binaryproto
		string vgg_net_path,   // vgg_net.txt
		string vgg_model_path // vgg.model
		);
	void Forward(const Dtype *image_data); // extract all features for all images in the batch
	bool GetFeatures(Dtype *fea, string layer_name);
	bool GetFeatures(Dtype *fea, const char *layer_name);

	int GetFeaDim(string layer_name); // get the feature dimension of the "layer_name" layer. layer_name can be "fc6", "fc7", "fc8", "prob"
	void SubMean(const Dtype *x, const Dtype *y, Dtype *dst);

	inline int BatchSize() { return batch_size_;  }// get the batch_size. The batch_size is equal to the third line in vgg_net.txt. batch_size == 1 by default
	inline int GetCropWidth() { return width_; }
	inline int GetCropHeight(){ return height_; }
	inline int GetResizeWidth() { return mean_width_;  }
	inline int GetResizeHeight() { return mean_height_;  }

private:
	boost::shared_ptr<Net<Dtype> > feature_extraction_net_;
	boost::shared_ptr<Blob<Dtype> > data_layer_;

	string data_layer_name_;  // data_layer_name_ == "data" by default
	Blob<Dtype> data_mean_;  // vgg_mean.binaryproto
	Blob<Dtype> data_sub_mean_;

	int batch_size_;
	int channels_;
	int width_;
	int height_;
	int count_;

	int mean_width_;
	int mean_height_;
};

#endif