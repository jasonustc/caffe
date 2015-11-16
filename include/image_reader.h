#ifndef IMAGE_READER_H_
#define IMAGE_READER_H_

#include <iostream>
#include <string>
using std::string;

template <typename Dtype>
class ImageReader
{
public:
	ImageReader();
	~ImageReader();
	bool ReadResizeImage(string image_path);
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
	Dtype *image_data_;
	int resize_width_;
	int resize_height_;
	int crop_width_;
	int crop_height_;
};


#endif