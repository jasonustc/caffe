#include "caffe/dmodel_loader.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"


template <typename Dtype>
DModelLoader<Dtype>::DModelLoader()
{
	data_layer_name_ = "data";
	
}

template <typename Dtype>
DModelLoader<Dtype>::~DModelLoader()
{
	std::cout << "model release" << std::endl;
}

//allow for both CPU and GPU mode
template <typename Dtype>
bool DModelLoader<Dtype>::LoadModel(string vgg_mean_path, string vgg_net_path, string vgg_model_path,
	Caffe::Brew device, const int device_id)
{
	//::google::InitGoogleLogging("extract vgg");
	if (device == Caffe::CPU){
		Caffe::set_mode(Caffe::CPU);
		LOG(INFO) << "Using CPU";
	}
	else{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(device_id);
		LOG(INFO) << "Using GPU " << device_id;
	}

	feature_extraction_net_ = boost::shared_ptr<Net<Dtype> >(new Net<Dtype>(vgg_net_path, caffe::TEST));
	feature_extraction_net_->CopyTrainedLayersFrom(vgg_model_path);

	caffe::BlobProto blob_proto;
	caffe::ReadProtoFromBinaryFileOrDie(vgg_mean_path.c_str(), &blob_proto);
	data_mean_.FromProto(blob_proto);

	if (!feature_extraction_net_->has_blob(data_layer_name_))
	{
		std::cout << "error, no data layer" << std::endl;
		return false;
	}

	//set the input of the net to data_layer_
	data_layer_ = feature_extraction_net_->blob_by_name(data_layer_name_);

	batch_size_ = data_layer_->num();
	channels_ = data_layer_->channels();
	width_ = data_layer_->width();
	height_ = data_layer_->height();
	count_ = batch_size_ * channels_ * width_ * height_;

	mean_width_ = data_mean_.width();
	mean_height_ = data_mean_.height();

	data_sub_mean_.Reshape(batch_size_, channels_, height_, width_);

	if (batch_size_ != 1)
	{
		std::cout << "error, batch_size should be equal to 1 (input_dim: 1)" << std::endl;
		return false;
	}

	return true;
}

template <typename Dtype>
void DModelLoader<Dtype>::SubMean(const Dtype *x, const Dtype *y, Dtype *dst)
{
	int img_index, top_index, mean_index;
	int h_off = (mean_height_ - height_) / 2;
	int w_off = (mean_width_ - width_) / 2;

	for (int h = 0; h < height_; ++h) 
	{
		img_index = h * width_ * channels_;
		for (int w = 0; w < width_; ++w) 
		{
			for (int c = 0; c < channels_; ++c) 
			{	
				top_index = (c * height_ + h) * width_ + w;
				mean_index = (c * mean_height_ + h_off + h) * mean_width_ + w_off + w;
				dst[top_index] = (x[img_index] - y[mean_index]);
				img_index++;
			}
		}
	}
}

//mean_width_: resize width, width_: width after crop
template <typename Dtype>
void DModelLoader<Dtype>::SubMean(const Dtype *x, const Dtype *y, Dtype *dst,
	int crop_type, bool mirror)
{
	int img_index, top_index, mean_index;
	int h_off;
	int w_off;
	switch (crop_type)
	{
	case 0:
		//left up
		h_off = 0;
		w_off = 0;
		break;
	case 1:
		//right up
		h_off = 0;
		w_off = mean_width_ - width_;
		break;
	case 2:
		//middle
		h_off = (mean_height_ - height_) / 2;
		w_off = (mean_width_ - width_) / 2;
		break;
	case 3:
		//left down
		h_off = mean_height_ - height_;
		w_off = 0;
		break;
	case 4:
		//right down
		h_off = mean_height_ - height_;
		w_off = mean_width_ - width_;
		break;
	default:
		LOG(FATAL) << "unkown crop type " << crop_type;
		break;
	}

	for (int h = 0; h < height_; ++h) 
	{
		img_index = h * width_ * channels_;
		for (int w = 0; w < width_; ++w) 
		{
			for (int c = 0; c < channels_; ++c) 
			{	
				if (mirror){
					top_index = (c * height_ + h) * width_ + (width_ - 1 - w);
				}
				else{
					top_index = (c * height_ + h) * width_ + w;
				}
				mean_index = (c * mean_height_ + h_off + h) * mean_width_ + w_off + w;
				dst[top_index] = (x[img_index] - y[mean_index]);
				img_index++;
			}
		}
	}
}

//TODO: refine to deal with multi-crops
template <typename Dtype>
void DModelLoader<Dtype>::Forward(const Dtype *image_data)
{
	std::vector<Blob<Dtype>*> input_vec;
	
	SubMean(image_data, data_mean_.cpu_data(), data_sub_mean_.mutable_cpu_data());

	memcpy(data_layer_->mutable_cpu_data(), data_sub_mean_.cpu_data(), sizeof(Dtype)* count_);
	feature_extraction_net_->Forward(input_vec);
}

template <typename Dtype>
void DModelLoader<Dtype>::Forward(const Dtype *image_data, int crop_type, bool mirror)
{
	std::vector<Blob<Dtype>*> input_vec;
	
	SubMean(image_data, data_mean_.cpu_data(), data_sub_mean_.mutable_cpu_data(),
		crop_type, mirror);

	memcpy(data_layer_->mutable_cpu_data(), data_sub_mean_.cpu_data(), sizeof(Dtype)* count_);
	feature_extraction_net_->Forward(input_vec);
}

template <typename Dtype>
bool DModelLoader<Dtype>::GetFeatures(Dtype *fea, const char *layer_name)
{
	if (!feature_extraction_net_->has_blob(layer_name))
	{
		std::cout << "error, Unknown layer_name " << layer_name << " in the network "<< std::endl;
		return false;
	}

	const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net_
		->blob_by_name(layer_name);

	int batch_size = feature_blob->num(); // batch_size must be 1
	int dim_features = feature_blob->count() / batch_size;

	const Dtype *feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(0);
	for (int i = 0; i < dim_features; i++)
	{
		fea[i] = feature_blob_data[i];
	}

	return true;
}

template <typename Dtype>
bool DModelLoader<Dtype>::GetFeatures(Dtype *fea, string layer_name)
{
	return GetFeatures(fea, layer_name.c_str());
}

template <typename Dtype>
int DModelLoader<Dtype>::GetFeaDim(string layer_name)
{
	if (!feature_extraction_net_->has_blob(layer_name))
	{
		std::cout << "error, Unknown layer_name " << layer_name << " in the network "<< std::endl;
		return -1;
	}

	const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net_
		->blob_by_name(layer_name);

	int batch_size = feature_blob->num();
	int dim_features = feature_blob->count() / batch_size;

	return dim_features;
}

template class DModelLoader<float>;
template class DModelLoader<double>;