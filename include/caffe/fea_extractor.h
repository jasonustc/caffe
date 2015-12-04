#ifndef FEA_EXTRACTOR_H_
#define FEA_EXTRACTOR_H_

#include "dmodel_loader.h"
#include "image_reader.h"

using boost::shared_ptr;

template <typename Dtype>
class FeaExtractor
{
public:
	FeaExtractor();
	~FeaExtractor();
	bool LoadModel(
		string mean_path,    // vgg_mean.binaryproto
		string net_path,    // vgg_net.txt
		string model_path, // vgg.model
		string layer_name,     // layer_name can be "fc6", "fc7", "fc8", "prob"
		Caffe::Brew device = Caffe::CPU, //GPU or CPU
		const int device_id = 0 //if GPU, device id
	);

	bool ExtractFea(string image_path); // extract vgg features
	/*
	 * For the model averaging task 
	 * @param crop_type: 0, left_up; 1, right_up; 2, middle; 3, left_down; 4, right_down
	 * @param mirror: true, do mirror transform; false, no mirror transform
	 */
	bool ExtractFea(string image_path, int crop_type, bool mirror);
	bool ExtractFea(void *scan0, int width, int height, int stride, int channel);
	bool ExtractCropFea(string image_path, const int dim);
	inline int GetFeaDim() { return dim_; }        // get the dimension of the features
	inline const Dtype* GetFea() const { return fea_.get(); } // get the pointer of vgg_features
	inline Dtype* GetMutableFea() { return fea_.get(); } // return mutable feature pointer

private:
	DModelLoader<Dtype> dmodel_loader_; // used for loading vgg model
	ImageReader<Dtype> image_reader_; // used for reading and cropping images

	string layer_name_;      // layer_name can be "fc6", "fc7", "fc8", "prob"
	int dim_;               // the dimension of the vgg_features
	boost::shared_ptr<Dtype> fea_;           // vgg_features
};

	bool CreateHandle(FeaExtractor<float>* &fea_extractor);

	 bool ReleaseHandle(FeaExtractor<float>* &fea_extractor);

	 bool LoadDModel(
		 void *fea_extractor,
		 string mean_path,    // mean.binaryproto
		 string net_path,    // net.txt
		 string model_path, // model
		 string layer_name,     // layer_name can be "fc6", "fc7", "fc8", "prob"
		 Caffe::Brew device = Caffe::CPU, //CPU or GPU
		 const int device_id = 0 // if GPU, device id
		);

	 bool ExtractFeaturesByPath(void *fea_extractor, string image_path);
	 bool ExtractFeaturesByPath(void *fea_extractor, string image_path,
		 const int crop_type, const bool mirror);
	 bool ExtractCropFeaturesByPath(void *fea_extractor, string image_path, const int dim);

	 bool ExtractFeaturesByData(
		void *fea_extractor, 
		void *scan0, 
		int image_width, 
		int image_height, 
		int image_stride, 
		int channel);

	 int GetFeaDim(void *fea_extractor);

	 const float* GetFeaPtr(void *fea_extractor);
	 float* GetMutableFeaPtr(void *fea_extractor);


#endif
