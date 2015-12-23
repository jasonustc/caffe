#include "caffe/fea_extractor.h"

bool CreateHandle(FeaExtractor<float>* &fea_extractor)
{
	fea_extractor = new FeaExtractor<float>();
	if (fea_extractor == NULL)
	{
		std::cout << "create handle error" << std::endl;
		return false;
	}
	return true;
}

bool ReleaseHandle(FeaExtractor<float>* &fea_extractor)
{
	if (fea_extractor != NULL)
	{
		delete fea_extractor;
		fea_extractor = NULL;
	}
	return true;
}

bool LoadDModel(
	void *fea_extractor,
	string mean_path,    // mean.binaryproto
	string net_path,    // net.txt
	string model_path, // model
	string layer_name,     // layer_name can be "fc6", "fc7", "fc8", "prob"
	Caffe::Brew device,
	const int device_id
	)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->LoadModel(mean_path, net_path, model_path, layer_name, device, device_id);
}

bool ExtractFeaturesByPath(void *fea_extractor, string image_path)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->ExtractFea(image_path);
}

bool ExtractCropFeaturesByPath(void *fea_extractor, string image_path, const int dim)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->ExtractCropFea(image_path, dim);
}

bool ExtractFeaturesByPath(void *fea_extractor, string image_path, 
	const int crop_type, const bool mirror)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->ExtractFea(image_path, crop_type, mirror);
}

bool ExtractFeaturesByData(
	void *fea_extractor, 
	void *scan0, 
	int image_width, 
	int image_height, 
	int image_stride, 
	int channel)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->ExtractFea(scan0, image_width, image_height,  image_stride, channel);
}

int GetFeaDim(void *fea_extractor)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->GetFeaDim();
}

const float* GetFeaPtr(void *fea_extractor)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->GetFea();
}

float* GetMutableFeaPtr(void *fea_extractor)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->GetMutableFea();
}

template <typename Dtype>
FeaExtractor<Dtype>::FeaExtractor()
{
//	fea_ = NULL;
	dim_ = -1;
}

template <typename Dtype>
FeaExtractor<Dtype>::~FeaExtractor()
{

}

template <typename Dtype>
bool FeaExtractor<Dtype>::LoadModel(
	string mean_path,    // vgg_mean.binaryproto
	string net_path,    // vgg_net.txt
	string model_path, // vgg.model
	string layer_name,     // layer_name can be "fc6", "fc7", "fc8", "prob"
	Caffe::Brew device,
	const int device_id
	)
{
	layer_name_ = layer_name;

	// load vgg model
	std::cout << "begin to load model" << std::endl;
	bool succ = dmodel_loader_.LoadModel(mean_path, net_path, model_path, device, device_id);
	if (!succ) 
	{ 
		std::cout << "error, failed to load model" << std::endl;
		return false; 
	}
	std::cout << "finish loading model" << std::endl;

	// get the feature dimension
	dim_ = dmodel_loader_.GetFeaDim(layer_name_);
	if (dim_ == -1)
	{
		std::cout << "error, failed to get feature dimension" << std::endl;
		return false; 
	}

	// allocate
	fea_ = boost::shared_ptr<Dtype>(new Dtype[dim_]);
	if (fea_ == NULL)
	{
		std::cout << "error, out of memory" << std::endl;
		return false;
	}

	int resize_width = dmodel_loader_.GetResizeWidth();
	int resize_height = dmodel_loader_.GetResizeHeight();
	image_reader_.SetResize(resize_width, resize_height);

	int crop_width = dmodel_loader_.GetCropWidth();
	int crop_height = dmodel_loader_.GetCropHeight();
	image_reader_.SetCropSize(crop_width, crop_height);

	return true;
}

template <typename Dtype>
bool FeaExtractor<Dtype>::ExtractFea(string image_path)
{
	bool succ = image_reader_.ReadResizeImage(image_path); // read and transform image
	if (!succ)
	{
		std::cout << "error, can not read image: " << image_path << std::endl;
		return false;
	}

	const Dtype *image_data = image_reader_.GetImgData();
	dmodel_loader_.Forward(image_data); // Forward and extract features

	succ = dmodel_loader_.GetFeatures(fea_.get(), layer_name_.c_str()); // copy the features to fea_ buffer
	if (!succ)
	{
		std::cout << "error, dmodel_loader_ failed to get features" << std::endl;
		return false;
	}

	return true;
}

template <typename Dtype>
bool FeaExtractor<Dtype>::ExtractFea(string image_path, int crop_type, bool mirror)
{
	bool succ = image_reader_.ReadResizeImage(image_path, crop_type, mirror); // read and transform image
	if (!succ)
	{
		std::cout << "error, can not read image: " << image_path << std::endl;
		return false;
	}

	const Dtype *image_data = image_reader_.GetImgData();
	dmodel_loader_.Forward(image_data, crop_type, mirror); // Forward and extract features

	succ = dmodel_loader_.GetFeatures(fea_.get(), layer_name_.c_str()); // copy the features to fea_ buffer
	if (!succ)
	{
		std::cout << "error, dmodel_loader_ failed to get features" << std::endl;
		return false;
	}

	return true;
}

//TODO: extract 10 crop feature in a time
//actually, we only need to read the image pixels once

template <typename Dtype>
void Average(const int dim, const int count, const Dtype* a, 
	const Dtype* b, Dtype* dst){
	for (int i = 0; i < dim; i++){
		dst[i] = (a[i]* count + b[i]) / (count + 1);
	}
}

template <typename Dtype>
bool FeaExtractor<Dtype>::ExtractCropFea(string image_path, const int dim)
{
	bool succ = image_reader_.ReadImage(image_path); // read image to buffer 
	if (!succ)
	{
		std::cout << "error, can not read image: " << image_path << std::endl;
		return false;
	}
	int count = 0;
	Dtype* avg_feat = new Dtype[dim];
	//average between 10 crops
	for (size_t c = 0; c < 5; c++){
		//mirror 
		image_reader_.GetCropImageData(c, true);
		const Dtype* image_data_1 = image_reader_.GetImgData();
		dmodel_loader_.Forward(image_data_1, c, true); // Forward and extract features
		succ = dmodel_loader_.GetFeatures(fea_.get(), layer_name_.c_str()); // copy the features to fea_ buffer
		if (!succ)
		{
			std::cout << "error, dmodel_loader_ failed to get features" << std::endl;
			return false;
		}
		Average(dim, count, avg_feat, fea_.get(), avg_feat);
		count++;

		//not mirror
		image_reader_.GetCropImageData(c, false);
		const Dtype* image_data_2 = image_reader_.GetImgData();
		dmodel_loader_.Forward(image_data_2, c, false); // Forward and extract features
		succ = dmodel_loader_.GetFeatures(fea_.get(), layer_name_.c_str()); // copy the features to fea_ buffer
		if (!succ)
		{
			std::cout << "error, dmodel_loader_ failed to get features" << std::endl;
			return false;
		}
		Average(dim, count, avg_feat, fea_.get(), avg_feat);
		count++;
	}

	memcpy(fea_.get(), avg_feat, sizeof(Dtype) * dim);
	return true;
}

template <typename Dtype>
bool FeaExtractor<Dtype>::ExtractFea(void *scan0, int width, int height, int stride, int channel)
{
	bool succ = image_reader_.ReadResizeImage(scan0, width, height, stride, channel); // read and transform image
	if (!succ)
	{
		std::cout << "error, can not read image: " << std::endl;
		return false;
	}

	const Dtype *image_data = image_reader_.GetImgData();
	dmodel_loader_.Forward(image_data); // Forward and extract features

	succ = dmodel_loader_.GetFeatures(fea_.get(), layer_name_.c_str()); // copy the features to fea_ buffer
	if (!succ)
	{
		std::cout << "error, dmodel_loader_ failed to get features" << std::endl;
		return false;
	}

	return true;
}
