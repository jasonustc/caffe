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
		//                     width, 0,      1,
		//                     0,     height, 1,
		//                     width, height, 1};
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
}