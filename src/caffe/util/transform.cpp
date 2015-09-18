//get more information from https://github.com/akanazawa/si-convnet
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "caffe/util/transform.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/blob.hpp"

using std::min;
using std::max;

namespace caffe{

	void TMatFromProto(const TransParameter &param, float *tmat, bool invert){
		//initialize to identity
		std::fill(tmat, tmat + 9, 0);
		tmat[0] = tmat[4] = tmat[9] = 1;
		//rotation
		if (param.rotation() != 0.){
			if (invert){
				AddRotation(-param.rotation(), tmat);
			}
			else{
				AddRotation(param.rotation(), tmat);
			}
		}
		//scale
		if (param.scale() != 1){
			CHECK(param.scale() > 0) << "Scale has to be >= 0" << param.scale();
			if (invert){
				AddScale(1. / param.scale(), tmat);
			}
			else{
				AddScale(param.scale(), tmat);
			}
		}
		//shift
		if (param.dx() != 0 || param.dy() != 0){
			if (invert){
				AddShift(-param.dx(), -param.dy(), tmat);
			}
			else{
				AddShift(param.dx(), param.dy(), tmat);
			}
		}
	}

	void AddScale(const float &scale, float *mat, const Direction dir){
		float tmp[9] = { scale, 0, 0, 0, scale, 0, 0, 0, 1 };
		AddTransform(mat, tmp, dir);
	}


	void AddRotation(const float &angle, float *mat, const Direction dir){
		//Angle in degrees
		float rad = angle * PI_F / 180;
		float tmp[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
		AddTransform(mat, tmp, dir);
	}

	void AddShift(const float &dx, const float &dy, float *mat, const Direction dir){
		//dx is width, dy is height
		float tmp[9] = { 1, 0, 0, 0, 1, 0, dx, dy, 1};
		AddTransform(mat, tmp, dir);
	}

	/*
	 *all the 2D transformations can be modeled by the combination of following 3 basic
	 *transformations
	 *rotation:
	 *                 [cos\theta  sin\theta  0]
	 *[x y 1] = [x y 1][-sin\theta cos\theta  0]
	 *                 [ 0            0       1]
	 *shift:
	 *                 [1  0  0]
	 *[x y 1] = [x y 1][0  1  0]
	 *                 [dx dy 1]
	 *scale:
	 *                 [s_x  0    0]
	 *[x y 1] = [x y 1][0    s_y  0]
	 *                 [0    0    1]
	 */
	void AddTransform(float *A, const float *B, const Direction dir){
		//matrix multiply A and B and store to A
		//i.e. A = A_copy * B + 0 * A
		//but gemm can't be done in inplace, so A has to be a copy of A
		//if dir == LEFT, does A = B * A_copy
		float A_copy[9];
		caffe_copy<float>(9, A, A_copy);
		dir == RIGHT ? caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.f,
			A_copy, B, 0.f, A) :
			caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.f,
			B, A_copy, 0.f, A);
	}
	
	//TODO(AJ): this is computing the offset. with affine transformation computing
	//the offset using the center of the new image is fine but not so for
	//perspective. What works for both cases is having offset as (min x, min y),
	//the left top corner of the image is the right offset.
	void GetNewSize(const int &height, const int &width, const float *mat,
		int &height_new, int &width_new){
		CHECK_GT(height, 0) << "height must larger than 0" << height;
		CHECK_GT(width, 0) << "width must larger than 0" << width;
		//4 corners
		// x, y, z
		// float corners[12] = {0,    0,      1,
		//                   width,   0,      1,
		//                     0,   height,   1,
		//                   width, height,   1};
		//in row, col, z
		//what does this mean?
		float corners[12] = { 0,            0,
			                  1,            static_cast<float>(height),
							  0,            1,
							  0,            static_cast<float>(width),
							  1,            static_cast<float>(height),
							  static_cast<float>(width), 1};
		float res[12];
		//Apply transformation: RIGHT multiply if using x, y, z corners, LEFT muliply
		//with y, x, z
		//res = 1 * corners x mat + 0.f * res
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 4, 3, 3, 1.f, corners, mat, 0.f, res);
		float max_col = max(max(res[1], res[4]), max(res[7], res[10]));
		float min_col = min(min(res[1], res[4]), min(res[7], res[10]));
		float max_row = max(max(res[0], res[3]), max(res[6], res[9]));
		float min_row = min(min(res[0], res[3]), min(res[6], res[9]));
		height_new = static_cast<int>(max_row - min_row);
		width_new = static_cast<int>(max_col - min_col);
	}

	//Following the inverse rule of 3x3 matrices using determinats
	//rewrites tmat into its inverse
	//TODO: convert tmat to double
	//use LU from lapack/blas if this is too numerically unstable?
	//invert index of images, image_transformed -> original_image
	void Invert3x3(float *A){
		float inv[9];
		//|A| = aei + bfg + cdh - (ceg + bdi + afh)
		//the determinant of matrix A
		/*A = [0 1 2]
		 *    [3 4 5]
		 *    [6 7 8]
		 *A^T = [0 3 6]      [+ - +]
		 *      [1 4 7] dot  [- + -]
		 *      [2 5 8]      [+ - +]
		 */
		float d1 = A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7];
		float d2 = A[0] * A[5] * A[7] + A[2] * A[4] * A[6] + A[8] * A[1] * A[3];
		float det = d1 - d2;
		printf("det: %.2f - %.2f = %.2f", d1, d2, det);
		CHECK_GT(det, 0);
		inv[0] = (A[8] * A[4] - A[5] * A[7]) / det;
		inv[1] = (A[7] * A[2] - A[1] * A[8]) / det;
		inv[2] = (A[1] * A[5] - A[4] * A[2]) / det;
		inv[3] = (A[6] * A[5] - A[3] * A[8]) / det;
		inv[4] = (A[0] * A[8] - A[6] * A[2]) / det;
		inv[5] = (A[3] * A[2] - A[0] * A[5]) / det;
		inv[6] = (A[3] * A[7] - A[6] * A[4]) / det;
		inv[7] = (A[6] * A[1] - A[0] * A[7]) / det;
		inv[8] = (A[0] * A[4] - A[3] * A[1]) / det;
		caffe_copy(9, inv, A);
	}

	//get the reflect location in matrix
	//just mirror pixels to the border
	template <typename Dtype> void Reflect(Dtype &val, const int size){
		if (val < 0.){
			val = -floor(val);
			val = static_cast<Dtype>(static_cast<int>(val) % (2 * size - 2));
		}
		if (val >= size){
			val = 2 * size - 2 - val;
		}
	}
	template void Reflect<float>(float &val, const int size);

	//if out of range, just repeat the pixels on the bord
	template <typename Dtype> void Clamp(Dtype &val, const int size){
		val = max(static_cast<Dtype>(0.), min(val, static_cast<Dtype>(size - 1)));
	}

	template void Clamp<float>(float &val, const int size);

	void get_transformed_coordinates(const int &height, const int width,
		float* tmat, int &height_new, int &width_new,
		float* &coord_data_res){
		//get new size with this transformation matrix
		GetNewSize(height, width, tmat, height_new, width_new);
		//invert the transformation matrix, this rewrites tmat to its inverse
		Invert3x3(tmat);
		//get identity indices
		//use heap b/c o.w. stack overflows with medium sized images.
		float *coord_data_tmp = new float[height_new * width_new * 3];
		GenBasicCoordMat(coord_data_tmp, width_new, height_new);
		//get centers
		float new_cy = static_cast<float>(height_new - 1) / 2.;
		float new_cx = static_cast<float>(width_new - 1) / 2.;

		//float old_cy = static_cast<float>(height)/2.;
		//float old_cx = static_cast<float>(width)/2.;

		//subtract the new center first
		//*this translation matrix has to be LEFT multied on tmat_inv*.
		//they have to be subtracted from coord before applying the rest of 
		//the transformation
		//note: this has to be swapped bc we're using [row, col, 1]
		//first move image to the center
		AddShift(-new_cy, -new_cx, tmat, LEFT);
		//apply transformation
		coord_data_res = new float[height_new * width_new * 3];
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, height_new* width_new, 3,
			3, 1.f, coord_data_tmp, tmat, 0.f, coord_data_res);
		delete[] coord_data_tmp;
	}

	void generate_nn_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float* coord_data_res,
		float* &coord_data){
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;
		//copy over results over after applying reflection dropping the 3rd dim
		//also need to add the old_center
		for (int ind = 0; ind < height_new * width_new; ++ind){
			float row = round(coord_data_res[3 * ind] + old_cy);
			float col = round(coord_data_res[3 * ind + 1] + old_cx);
			switch (border)
			{
			case CROP:
				if ((row >= height || row < 0) || (col >= width) || (col < 0)){
					coord_data[ind] = -1;
					continue;
				}
				break;
			case CLAMP:
				Clamp(row, height);
				Clamp(col, width);
				break;
			case REFLECT:
				Reflect(row, height);
				Reflect(col, width);
				break;
			default:
				LOG(FATAL) << "Unkown border mode.";
				break;
			}
			//save index
			coord_data[ind] = round(row) * width + round(col);
		}
	}

	/*
	 *Going to save [ind(p00), ind(p11), dc(x-x0), dr(y-y0)] for each pixel 
	 *in the new image
	 *These will be stored in 4xHxW matrix read in columnwise manner i.e.
	 *coord = [ind_0(p00), ind_1(p00),...
	 *         ind_0(p11), ind_1(p11),...
	 *         dc_0, dc_1, ...
	 *         dr_0, dr_1, ...]
	 * bc so that the first row is the NN-coord
	 *& compatible with max-pooling range checking later(see MaxTransSetSwitch).
	 */
	void generate_bilinear_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float* coord_data_res,
		float* &coord_data){
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;

		//copy over results over after applying reflection dropping the 3rd
		//also need to add the old_center
		//first remove the influence of the center, then resume to the old center
		float row, col, row0, col0, row1, col1, dc, dr;
		int N = height_new * width_new;
		for (int ind = 0; ind < N; ++ind){
			row = coord_data_res[3 * ind] + old_cy;
			col = coord_data_res[3 * ind] + old_cx;
			//p00=>(r0, c0) p11=>(r1, c1)
			//save index
			switch (border)
			{
			case CROP:
				if ((row >= height - 0.5 || row < -0.5) || 
					(col >= width - 0.5 || col < -0.5)){
					coord_data[ind] = -1;
					continue;
				}
				break;
			case CLAMP:
				Clamp(row, height);
				Clamp(row, height);
				break;
			case REFLECT:
				Reflect(row, height);
				Reflect(col, width);
				break;
			default:
				LOG(FATAL) << "Unknown border mode.";
				break;
			}
			//p00
			//round towards zero, the magnitude should not larger than x.
			row0 = trunc(row);
			col0 = trunc(col);
			//p11
			row1 = trunc(row + 1) > (height - 1) ? height - 1 : trunc(row + 1);
			col1 = trunc(col + 1) > (height - 1) ? height - 1 : trunc(col + 1);

			//if p00 is outside, don't compute difference
			dc = col0 == col1 ? 0 : col - col0;
			dr = row0 == row1 ? 0 : row - row0;
			CHECK(dc >= 0) << "dc has to be pos " << dc;
			CHECK(dr >= 0) << "dr has to be pos " << dr;
			coord_data[ind] = row0 * width + col0;
			coord_data[ind + N] = row1 * width + col1;
			coord_data[ind + 2 * N] = dc;
			coord_data[ind + 3 * N] = dr;
		}
	}

	//This doesn't change the size of the input
	void GenCoordMatCrop(const float &scale, const float &rotation,
		const int &height, const int &width, Blob<float> *coord,
		const Border &border, const Interp &interp){
		//0. make tmat
		float tmat[9];
		std::fill(tmat, tmat + 9, 0);
		tmat[0] = tmat[4] = tmat[8] = 1;
		AddScale(scale, tmat);
		AddRotation(rotation, tmat);

		Invert3x3(tmat);

		float cy = static_cast<float>(height - 1) / 2.;
		float cx = static_cast<float>(width - 1) / 2.;

		//substract center:
		AddShift(-cy, -cx, tmat, LEFT);

		float *coord_data_tmp = new float[height * width * 3];
		//give ind, we can find the col and row index
		GenBasicCoordMat(coord_data_tmp, width, height);
		//Apply transformation
		float *coord_data_res = new float[height * width * 3];

		caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, height * width, 3, 3, 1.f,
			coord_data_tmp, tmat, 0.f, coord_data_res);

		float *coord_data;
		switch (interp)
		{
		case NN:
			coord->Reshape(1, 1, height * width, 1);
			coord_data = coord->mutable_cpu_data();
			generate_nn_coord(height, width, height, width, border, coord_data_res,
				coord_data);
		case BILINEAR:
			coord->Reshape(1, 1, height * width * 4, 1);
			coord_data = coord->mutable_cpu_data();
			generate_bilinear_coord(height, width, height, width, border,
				coord_data_res, coord_data);
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
		//clean up
		delete[] coord_data_tmp;
		delete[] coord_data_res;
	}

	void GenCoordMat(float *tmat, const int &height, const int &width,
		Blob<float>* coord, int &height_new, int &width_new,
		const Border &border, const Interp &interp){
		float *coord_data_res = NULL;
		get_transformed_coordinates(height, width, tmat, height_new, width_new, 
			coord_data_res);
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;

		float* coord_data;
		switch (interp)
		{
		case NN:
			//set the size of coord blob
			coord->Reshape(1, 1, height_new * width_new * 4, 1);
			coord_data = coord->mutable_cpu_data();
			generate_nn_coord(height, width, height_new, width_new, border,
				coord_data_res, coord_data);
			break;
		case BILINEAR:
			coord->Reshape(1, 1, height_new * width_new * 4, 1);
			coord_data = coord->mutable_cpu_data();
			generate_bilinear_coord(height, width, height_new, width_new, border,
				coord_data_res, coord_data);
		default:
			LOG(ERROR) << "Unknown interpolation mode " << interp;
			break;
		}
		//clean up
		delete[] coord_data_res;
	}

	//fills width * height by 3 matrix that holds identity homogeneous coordinates
	//get mat index
	void GenBasicCoordMat(float* coord, const int &width, const int &height){
		int row, col;
		for (int ind = 0; ind < width * height; ++ind){
			//compute subscripts from this index
			row = ind / width;
			col = ind % width;

			coord[3 * ind] = row;
			coord[3 * ind + 1] = col;
			coord[3 * ind + 2] = 1;
		}
	}

	//fills width * height x 1 vector with the identity indices
	void GenBasicCoordInds(const int &width, const int &height,
		Blob<float>* coord){
		coord->Reshape(1, 1, width*height, 1);
		float* coord_data = coord->mutable_cpu_data();
		int row, col;
		for (int ind = 0; ind < width * height; ++ind){
			//compute subscripts from this index
			row = ind / width;
			col = ind % width;
			coord_data[ind] = row * width + col;
		}
	}

	// TODO(AJ): Move this comment elsewhere.
	// For downpooling, when undoing all the transformation, need to only return the
	// pixels that correspond to the canonical response. If the output is smaller
	// than the canonical output size, then the output is padded with 0 to fit the
	// size. If the output is larger, then the crop of size canonical output is
	// taken at the center.
	// For odd canonical output sizes, the extra pixel has to happen to the right of
	// the center, i.e. Crop an 4x4 into 3x3 at the center can be either (in 1-D)
	// indices [0, 1, 2] or [1, 2, 3]. The correct indices in SICNN case is [1, 2,
	// 3] because the first few convolution outputs don't have corresponding centers
	// as the output of the canonical size.
	// Assumes that coord_new is length target_w*target_h.
	// Flow:
	// up_layer & tied conv: applies T to an image (img*T), then convolve the
	// transformed image (img*T*W)
	// downpool_layer: applies T^{-1}, which has a different size than the canonical
	// size (unless stride==kernel size0
	// if img*T*W*T^{-1} is smaller than img*W (when image was down-sampled), then
	// pad the border with 0 (i.e. coordinate points to -1).
	// else, the corresponding portion is the center of the bigger response, so crop
	// at the center.
	void CropCenter(const float *coord, const ImageSize &original,
		const ImageSize &target, const Interp &interp,
		float* coord_new){
		int new_start_r = 0, new_start_c = 0;
		int old_start_r = 0, old_start_c = 0;
		int match_width = 0, match_height = 0;
		//find the start and row of the larger dimension.

		if (original.height < target.height){//pad
			new_start_r = ceil((target.height - original.height) / 2.);
			match_height = original.height;
		}
		else{//crop center
			old_start_r = ceil((original.height - target.height) / 2.);
			match_height = target.height;
		}
		if (original.width < target.width){//pad
			new_start_c = ceil((target.width - original.width) / 2.);
			match_width = original.width;
		}
		else{//crop center
			new_start_c = ceil((original.width - target.width) / 2.);
			match_width = target.width;
		}
		switch (interp)
		{
		case NN:
		{
		   //nn_crop_center(coord, original, target, coord_new);
		   std::fill(coord_new, coord_new + target.height * target.width, -1);
		   for (int i = 0, row_new = new_start_r, row_old = old_start_r; i < match_height;
			   ++i, ++row_new, ++row_old){
			   for (int j = 0, col_new = new_start_c, col_old = old_start_r; j < match_width;
				   ++j, ++col_new, ++col_old){
				   const int ind_new = row_new * target.width + col_new;
				   const int ind_old = row_old * original.width + col_old;
				   coord_new[ind_new] = coord[ind_old];
			   }
		   }
		}
		break;
		case BILINEAR:
		{
			//bilnear_crop_center(coord, original, target, coord_new);
			//maybe pixel value of (i, j) is determined by (i-1, j),(i+1, j),(i, j-1) and (i, j+1)?
			 int dim_old = original.width * original.height;
			 int dim_new = target.width * target.height;
			 //why 4 here?
			 std::fill(coord_new, coord_new + 4 * dim_new, -1);
			 for (int i = 0, row_new = new_start_r, row_old = old_start_r; i < match_height;
				 ++i, ++row_new, ++row_old){
				 for (int j = 0, col_new = new_start_c, col_old = old_start_c; j < match_width;
					 ++j, ++col_new, ++col_old){
					 const int ind_new = row_new * target.width + col_new;
					 const int ind_old = row_old * original.width + col_old;
					 coord_new[ind_new] = coord[ind_old];
					 coord_new[ind_new + dim_new] = coord[ind_old + dim_old];
					 coord_new[ind_new + 2 * dim_new] = coord[ind_old + 2 * dim_old];
					 coord_new[ind_new + 3 * dim_new] = coord[ind_old + 3 * dim_old];
				 }
			 }
		}
		break;
		default:
			LOG(ERROR) << "Unkown interpolate mode" << interp;
			break;
		}
	}

	template <typename Dtype>
	void InterpImageNN_cpu(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped, const Interp &interp){
		switch (interp)
		{
		case NN:
			nn_interpolation(orig, coord, warped);
			break;
		case BILINEAR:
			bilinear_interpolation(orig, coord, warped);
		default:
			LOG(ERROR) << "Unkown interpolation mode" << interp;
			break;
		}
	}

	template void InterpImageNN_cpu(const Blob<float>* orig, const float* coord,
		Blob<float>* warped, const Interp &interp);

	//just like crop
	template <typename Dtype>
	void nn_interpolation(const Blob<Dtype>* &orig, const float* &coord,
		Blob<Dtype>* &warped){
		//Get the parameters from the original and warped and apply the
		//transformation to it.
		int ind_warped, ind_orig, h_orig, w_orig;
		int width_orig = orig->width();
		int height_orig = orig->height();
		int num = warped->num();
		int channels = warped->channels();
		int height = warped->height();
		int width = warped->width();

		const Dtype *orig_data = orig->cpu_data();
		Dtype* warped_data = warped->mutable_cpu_data();

		for (int n = 0; n < num; ++n){//for each img
			for (int c = 0; c < channels; ++c){//for each channel
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_warped = h * width + w;//index in warped image
						ind_orig = static_cast<int>(coord[ind_warped]);
						if (ind_orig >= 0){ //do only if valid index
							h_orig = ind_orig / width_orig; //row in original
							w_orig = ind_orig % width_orig; //col in original
							warped_data[((n * channels + c) * height + h) * width + w] =
								orig_data[((n * channels + c) * height_orig) * width_orig + w_orig];
						}
						else{
							warped_data[((n*channels + c) * height + h) * width + w] = 0;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void bilinear_interpolation(const Blob<Dtype>* &orig, const float* &coord,
		Blob<Dtype>* &warped){
		//Get the parameters from the original and warped and apply the
		//transformation to it.
		int ind_warped, ind_orig, r0, c0, r1, c1, ind_p11;
		int dc, dr, w00, w01, w10, w11;
		Dtype p00, p01, p10, p11;
		int width_orig = orig->width();
		int height_orig = orig->height();
		int num = warped->num();
		int channels = warped->channels();
		int height = warped->height();
		int width = warped->width();
		int N = width * height;
		const Dtype *orig_data = orig->cpu_data();
		Dtype* warped_data = warped->mutable_cpu_data();

		for (int n = 0; n < num; ++n){//for each image
			for (int c = 0; c < channels; ++c){//for each channel
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_warped = h * width + w;//index for coordinate in new image
						ind_orig = static_cast<int>(coord[ind_warped]);//p00
						if (ind_orig > 0){//do only if p00 is valid index
							r0 = ind_orig / width_orig;
							c0 = ind_orig % width_orig;
							//Coordinates are stored as 4 x N matrix
							ind_p11 = static_cast<int>(coord[ind_warped + N]);
							r1 = ind_p11 / width;
							c1 = ind_p11 % width;

							dc = coord[ind_warped + 2 * N];
							dr = coord[ind_warped + 3 * N];

							//bilinear interpolation
							//f(x,y) \approx f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
							w00 = (1 - dc) * (1 - dr);
							w01 = (1 - dr) * dc;
							w10 = (1 - dc) * dr;
							w11 = dr * dc;

							int offset = (n * channels + c) * height_orig;
							p00 = orig_data[(offset + r0) * width_orig + c0];
							p01 = orig_data[(offset + r0) * width_orig + c1];
							p10 = orig_data[(offset + r1) * width_orig + c0];
							p11 = orig_data[(offset + r1) * width_orig + c1];

							warped_data[((n*channels + c) * height + h) * width + w] =
								w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11;
						}
						else{
							warped_data[((n*channels + c) * height + h) * width + w] = 0;
						}
					}
				}
			}
		}
	}
	template void bilinear_interpolation(const Blob<float>* &orig, const float* &coord,
		Blob<float>* &warped);

	template <typename Dtype>
	void PropagateErrorNN_cpu(const Blob<Dtype>* top, const float* coord,
		Blob<Dtype>* bottom, const Interp &interp){
		switch (interp)
		{
		case NN:
			nn_propagation(top, coord, bottom);
			break;
		case BILINEAR:
			bilinear_interpolation(top, coord, bottom);
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
	}

	//Explicit instantiation
	template void PropagateErrorNN_cpu(const Blob<float>* top, const float* coord,
		Blob<float>* bottom, const Interp &interp);

	template <typename Dtype>
	void nn_propagation(const Blob<Dtype>* & top, const float* &coord,
		Blob<Dtype>* &bottom){
		//I will simply take the error at each location in top and add it to the 
		//corresponding neuron
		//in the bottom blob based on the coord indices
		//IMP: IT IS ASSUMED THAT THE BOTTOM DIFF IS PROPERLY PRE-INITIALIZED
		//I.E. HAS ALL ZEROS OR PROPER ERROR VALUES
		int ind_top, ind_bottom, h_bottom, w_bottom;

		int num = top->num();
		int channels = top->channels();
		int height = top->height();
		int width = top->width();

		int height_bottom = bottom->height();
		int width_bottom = bottom->width();

		const Dtype *top_diff = top->cpu_diff();
		Dtype* bottom_diff = bottom->mutable_cpu_diff();

		//Loop over top locations
		for (int n = 0; n < num; ++n){
			for (int c = 0; c < channels; ++c){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_top = h*width + w;
						ind_bottom = static_cast<int>(coord[ind_top]);
						if (ind_bottom > 0){//do only for valid index
							h_bottom = ind_bottom / width_bottom; //row
							w_bottom = ind_bottom % width_bottom; //col
							bottom_diff[((n* channels + c) * height_bottom + h_bottom) * width_bottom
								+ w_bottom] +=
								top_diff[((n*channels + c) * height + h) * width + w];
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void bilinear_propagation(const Blob<Dtype>* & top, const float* & coord,
		Blob<Dtype>* &bottom){
		//AJ: Just like forward image interpolation
		//IMP: IT IS ASSUMED THAT THE BOTTOM IS PROPERLY PRE_INITIALIZED AND HAS ALL
		//ZEROS OR PROPER ERROR VALUES
		int ind_top, ind_bottom, r0, c0, ind_p11, r1, c1;
		float dc, dr, w00, w01, w10, w11;
		int num = top->num();
		int channels = top->channels();
		int height = top->height();
		int width = top->width();

		int N = width* height;

		int height_bottom = bottom->height();
		int width_bottom = bottom->width();

		const Dtype *top_diff = top->cpu_diff();
		Dtype* bottom_diff = bottom->mutable_cpu_diff();

		//Loop over top locations
		for (int n = 0; n < num; ++n){
			for (int c = 0; c < channels; ++c){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_top = h * width + w;
						ind_bottom = static_cast<int>(coord[ind_top]);
						if (ind_bottom > 0){ //do only if valid index
							r0 = ind_bottom / width_bottom; //row
							c0 = ind_bottom % width_bottom; //col

							//Coordinates are stores as 4 x N matrix
							ind_p11 = static_cast<int>(coord[ind_top + N]);
							r1 = ind_p11 / width_bottom;
							c1 = ind_p11 % width_bottom;

							dc = coord[ind_top + 2 * N];
							dr = coord[ind_top + 3 * N];

							w00 = (1 - dc)*(1 - dc);
							w01 = (1 - dr)*dc;
							w10 = (1 - dc)*dr;
							w11 = dr * dc;

							int offset = (n * channels + c)*height_bottom;

							float top_error =
								top_diff[((n*channels + c) * height + h) * width + w];

							//propagate error after weighting with its bilinear coefficients
							//p00
							bottom_diff[(offset + r0) * width_bottom + c0] += w00 * top_error;
							//p01
							bottom_diff[(offset + r0) * width_bottom + c1] += w01 * top_error;
							//p10
							bottom_diff[(offset + r1) * width_bottom + c0] += w10 * top_error;
							//p11
							bottom_diff[(offset + r1) * width_bottom + c1] += w11 * top_error;
						}//end if index if valid
					}
				}
			}
		}
	}
}