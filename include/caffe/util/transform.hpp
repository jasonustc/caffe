//Please refer to https://github.com/akanazawa/si-convnet for more details

#ifndef CAFFE_UTIL_TRANSF_H_
#define CAFFE_UTIL_TRANSF_H_

#include <limits>
#include "caffe/blob.hpp"

namespace caffe{
	const float PI_F = 3.14159265358979f;
	enum Direction {RIGHT, LEFT};

	//Crop is zero-padding, CLAMP is border replicate, REFLECT is mirror.
	enum Border {CROP, CLAMP, REFLECT};
	enum Interp {NN, BILINEAR};

	struct ImageSize
	{
		ImageSize():width(0), height(0){};
		ImageSize(const int width, const int height) : width(width), height(height){};
		int width;
		int height;
	};

	//compute parameters in transformation matrix by given transform type
	void TMatFromParam(const int trans_type, const float param1, const float param2, float *tmat, bool invert = false);

	//matrix is multiplied to the existing one from the right
	void AddRotation(const float &angle, float *mat, const Direction dir = RIGHT);
	void AddScale(const float &scale, float* mat, const Direction dir = RIGHT);
	void AddShift(const float &dx, const float& dy, float *mat, const Direction dir = RIGHT);

	//m = m * t
	void AddTransform(float *mat, const float *tmp, const Direction dir = RIGHT);

	void Invert3x3(float *A);

	void generate_nn_coord(const int &height, const int &width,
		const int &height_new,const int &width_new, const Border &border, 
		const float* coord_data_res, float* &coord_data);

	void generate_bilinear_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float *coord_data_res,
		float *&coord_data);

	//This one doesn't change the size. Used in jittering the data with scale/rotation.
	void GenCoordMatCrop(float* tmat,
		const int &height, const int &width, Blob<float>& ori_coord,
		Blob<float>& coord_idx, const Border &border = CROP, const Interp &interp = NN);

	//Generates identity coordinates.
	void GenBasicCoordMat(float *coord, const int &width, const int &height);
	//identity coordinates in indices
	void GenBasicCoordMatInds(const int &width, const int &height,
		Blob<float> *coord);

	template <typename Dtype> void Reflect(Dtype &val, const int size);
	template <typename Dtype> void Clamp(Dtype &val, const int size);

	template <typename Dtype>
	void InterpImageNN_cpu(const Blob<Dtype> *orig, const float *coord,
		Blob<Dtype> *warped, const Interp &interp = NN);

	template <typename Dtype>
	void nn_interpolation(const Blob<Dtype> *&orig, const float *&coord,
		Blob<Dtype> *&warped);

	template <typename Dtype>
	void bilinear_interpolation(const Blob<Dtype> *&orig, const float *&coord,
		Blob<Dtype> *&warped);

	template <typename Dtype>
	void BackPropagateErrorNN_cpu(const Blob<Dtype> *top, const float *coord,
		Blob<Dtype> *bottom, const Interp &interp = NN);

	template <typename Dtype>
	void nn_backpropagation(const Blob<Dtype> *&top, const float *&coord,
		Blob<Dtype>* &bottom);

	template <typename Dtype>
	void bilinear_backpropagation(const Blob<Dtype>* & top, const float* & coord,
		Blob<Dtype>* &bottom);

	//gpu functions
	template <typename Dtype>
	void InterpImageNN_gpu(const Blob<Dtype> *orig, const float *coord,
		Blob<Dtype> *warped, const Interp &interp = NN);

	template <typename Dtype>
	void BackPropagateErrorNN_gpu(const Blob<Dtype> *top, const float *coord,
		Blob<Dtype> *bottom, const Interp &interp = NN);

}//namespace caffe

#endif