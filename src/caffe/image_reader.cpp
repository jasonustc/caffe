#include "caffe/image_reader.h"
#include <opencv\highgui.h>
#include <fstream>
#include "caffe/common.hpp"

template <typename Dtype>
ImageReader<Dtype>::ImageReader()
{
	image_data_ = NULL;
	resize_height_ = 256;
	resize_width_ = 256;
}

template <typename Dtype>
ImageReader<Dtype>::~ImageReader()
{
	if (image_data_ != NULL)
	{
		delete []image_data_;
		image_data_ = NULL;
	}
}

template <typename Dtype>
void ImageReader<Dtype>::Print()
{
	std::ofstream outfile("D:/highlight_code/code/caffe-windows_qing_D/deploy/vgg/img.txt");
	for (int h = 0; h < crop_height_; h++)
	{
		for (int w = 0; w < crop_width_ * 3; w++)
		{
			outfile << image_data_[h * crop_width_ * 3 + w] << " ";
		}
		outfile << std::endl;
	}
}

template <typename Dtype>
bool ImageReader<Dtype>::ReadImage(string image_path){
//	cv::Mat cv_img;
	cv::Mat cv_img_origin = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

	if (!cv_img_origin.data) 
	{
		std::cout << "Could not open or find file " << image_path << std::endl;
		return false;
	}
	if (cv_img_origin.channels() != 3)
	{
		std::cout << "error, the channel of image must be 3" << std::endl;
		return false;
	}

	if (resize_width_ <= 0 && resize_height_ <= 0)
	{
		std::cout << "resize_width_ or resize_height_ must > 0" << std::endl;
		return false;
	}
	cv::resize(cv_img_origin, cv_ori_img_, cv::Size(resize_width_, resize_height_));
	return true;
}

template <typename Dtype>
bool ImageReader<Dtype>::GetCropImageData(const int crop_type, const bool mirror){
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
		w_off = resize_width_ - crop_width_;
		break;
	case 2:
		//middle
		h_off = (resize_height_ - crop_height_) / 2;
		w_off = (resize_width_ - crop_width_) / 2;
		break;
	case 3:
		//left down
		h_off = resize_height_ - crop_height_;
		w_off = 0;
		break;
	case 4:
		//right down
		h_off = resize_height_ - crop_height_;
		w_off = resize_width_ - crop_width_;
		break;
	default:
		LOG(FATAL) << "unkown crop type " << crop_type;
		break;
	}
	cv::Rect roi(w_off, h_off, crop_width_, crop_height_);
	cv::Mat cv_cropped_img = cv_ori_img_(roi);

	for (int h = 0; h < crop_height_; h++)
	{
		uchar *ptr = cv_cropped_img.ptr<uchar>(h);
		for (int w = 0; w < crop_width_ * 3; w++)
		{
			image_data_[h * crop_width_ * 3 + w] = static_cast<Dtype>(ptr[w]);
		}
	}

	return true;
}

template <typename Dtype>
bool ImageReader<Dtype>::ReadResizeImage(string image_path)
{
	cv::Mat cv_img;
	cv::Mat cv_img_origin = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

	if (!cv_img_origin.data) 
	{
		std::cout << "Could not open or find file " << image_path << std::endl;
		return false;
	}
	if (cv_img_origin.channels() != 3)
	{
		std::cout << "error, the channel of image must be 3" << std::endl;
		return false;
	}

	if (resize_width_ <= 0 && resize_height_ <= 0)
	{
		std::cout << "resize_width_ or resize_height_ must > 0" << std::endl;
		return false;
	}
	cv::resize(cv_img_origin, cv_img, cv::Size(resize_width_, resize_height_));

	int h_off = (resize_height_ - crop_height_) / 2;
	int w_off = (resize_width_ - crop_width_) / 2;
	//here just use the middle crop for feature extraction
	cv::Rect roi(w_off, h_off, crop_width_, crop_height_);
	cv::Mat cv_cropped_img = cv_img(roi);

	for (int h = 0; h < crop_height_; h++)
	{
		uchar *ptr = cv_cropped_img.ptr<uchar>(h);
		for (int w = 0; w < crop_width_ * 3; w++)
		{
			image_data_[h * crop_width_ * 3 + w] = static_cast<Dtype>(ptr[w]);
		}
	}

	return true;
}

template <typename Dtype>
bool ImageReader<Dtype>::ReadResizeImage(string image_path, int crop_type, bool mirror)
{
	cv::Mat cv_img;
	cv::Mat cv_img_origin = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

	if (!cv_img_origin.data) 
	{
		std::cout << "Could not open or find file " << image_path << std::endl;
		return false;
	}
	if (cv_img_origin.channels() != 3)
	{
		std::cout << "error, the channel of image must be 3" << std::endl;
		return false;
	}

	if (resize_width_ <= 0 && resize_height_ <= 0)
	{
		std::cout << "resize_width_ or resize_height_ must > 0" << std::endl;
		return false;
	}
	cv::resize(cv_img_origin, cv_img, cv::Size(resize_width_, resize_height_));

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
		w_off = resize_width_ - crop_width_;
		break;
	case 2:
		//middle
		h_off = (resize_height_ - crop_height_) / 2;
		w_off = (resize_width_ - crop_width_) / 2;
		break;
	case 3:
		//left down
		h_off = resize_height_ - crop_height_;
		w_off = 0;
		break;
	case 4:
		//right down
		h_off = resize_height_ - crop_height_;
		w_off = resize_width_ - crop_width_;
		break;
	default:
		LOG(FATAL) << "unkown crop type " << crop_type;
		break;
	}
	cv::Rect roi(w_off, h_off, crop_width_, crop_height_);
	cv::Mat cv_cropped_img = cv_img(roi);

	for (int h = 0; h < crop_height_; h++)
	{
		uchar *ptr = cv_cropped_img.ptr<uchar>(h);
		for (int w = 0; w < crop_width_ * 3; w++)
		{
			image_data_[h * crop_width_ * 3 + w] = static_cast<Dtype>(ptr[w]);
		}
	}

	return true;
}

template <typename Dtype>
bool ImageReader<Dtype>::ReadResizeImage(
	void *scan0,
	int width,
	int height,
	int stride,
	int channel)
{
	cv::Mat cv_img;
	cv::Mat cv_img_origin(height, width, CV_8UC3, scan0, stride);
	
	if (!cv_img_origin.data || scan0 == NULL)
	{
		std::cout << "Could not open or find file " << std::endl;
		return false;
	}
	if (cv_img_origin.channels() != 3)
	{
		std::cout << "error, the channel of image must be 3" << std::endl;
		return false;
	}

	if (resize_width_ <= 0 && resize_height_ <= 0)
	{
		std::cout << "resize_width_ or resize_height_ must > 0" << std::endl;
		return false;
	}

	cv::resize(cv_img_origin, cv_img, cv::Size(resize_width_, resize_height_));

	int h_off = (resize_height_ - crop_height_) / 2;
	int w_off = (resize_width_ - crop_width_) / 2;
	cv::Rect roi(w_off, h_off, crop_width_, crop_height_);
	cv::Mat cv_cropped_img = cv_img(roi);

	for (int h = 0; h < crop_height_; h++)
	{
		uchar *ptr = cv_cropped_img.ptr<uchar>(h);
		for (int w = 0; w < crop_width_ * 3; w++)
		{
			image_data_[h * crop_width_ * 3 + w] = static_cast<Dtype>(ptr[w]);
		}
	}

	return true;
}

template class ImageReader<float>;
template class ImageReader<double>;
