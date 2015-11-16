#include "image_reader.h"
#include <opencv\highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

template <typename Dtype>
ImageReader<Dtype>::ImageReader()
{
	image_data_ = NULL;
	resize_height_ = 255;
	resize_width_ = 255;
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
