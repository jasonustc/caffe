#ifndef IMAGE_READER_H_
#define IMAGE_READER_H_

#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;

template <typename Dtype>
class ImageReader
{
public:
	ImageReader();
	~ImageReader();
	bool ReadImage(string image_path);
	/*
	 * For the model averaging task 
	 * @param crop_type: 0, left_up; 1, right_up; 2, middle; 3, left_down; 4, right_down
	 * @param mirror: true, do mirror transform; false, no mirror transform
	 */
	bool GetCropImageData(const int crop_type, const bool mirror);
	bool ReadResizeImage(string image_path);
	/*
	 * For the model averaging task 
	 * @param crop_type: 0, left_up; 1, right_up; 2, middle; 3, left_down; 4, right_down
	 * @param mirror: true, do mirror transform; false, no mirror transform
	 */
	bool ReadResizeImage(string image_path, int crop_type, bool mirror);
	bool ReadResizeImage(
		void *scan0,
		int width,
		int height,
		int stride,
		int channel);

	inline const Dtype *GetImgData() const { return image_data_; }
	inline void SetResize(int resize_width, int resize_height){ resize_width_ = resize_width; resize_height_ = resize_height; }
	inline void SetCropSize(int crop_width, int crop_height)
	{
		crop_width_ = crop_width;
		crop_height_ = crop_height;
		image_data_ = new Dtype[3 * crop_width_ * crop_height_];
	}
	void Print();

private:
	//image data after crop/mirror
	Dtype *image_data_;
	//image data before crop
	cv::Mat cv_ori_img_;
	int resize_width_;
	int resize_height_;
	int crop_width_;
	int crop_height_;
};


#endif