//Implemented by Angjoo Kanzawa, Abhishek Sharma 2013
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <thrust\device_vector.h>
#include <thrust\count.h>

#include "caffe/common.hpp"
#include "caffe/util/transform.hpp"
#include "caffe/blob.hpp"

namespace caffe{
	template <typename Dtype>
	__global__ void nn_interpolation_kernel(const int nthreads, const Dtype *oldDPtr,
		const int oldSheetCount, Dtype* newDPtr,
		const int newSheetCount, const float* coord){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			int offset = index % newSheetCount;
			int numSheet = index / newSheetCount;
			int backSheetOffset = static_cast<int>(coord[offset]);
			if (backSheetOffset >= 0){
				newDPtr[numSheet * newSheetCount + offset] =
					oldDPtr[numSheet * oldSheetCount + backSheetOffset];
			}
			else{
				newDPtr[numSheet * newSheetCount + offset] = 0;
			}
		}
	}

	template <typename Dtype>
	__global__ void bilinear_interpolation_kernel(const int nthreads, const Dtype* oldDPtr,
		const int oldSheetCount, Dtype* newDPtr, const int newSheetCount, const float* coord,
		const int W){
		//need W
		//what is W here? maybe the width of transformed matrix?
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			int offset = index % newSheetCount; //p00: r0 * W + c0
			int numSheet = index / newSheetCount;
			int backSheetOffset = static_cast<int>(coord[offset]);
			if (backSheetOffset >= 0){
				int c0 = backSheetOffset % W;
				//p11: r1 * W + c1
				int ind_p11 = static_cast<int>(coord[offset + newSheetCount]);
				int c1 = ind_p11 % W;

				int ind_p01 = backSheetOffset - c0 + c1;//r0 * W + c1
				int ind_p10 = ind_p11 - c1 + c0; //r1 * W + c0

				float dc = coord[offset + 2 * newSheetCount];
				float dr = coord[offset + 3 * newSheetCount];

				float w00 = (1 - dc)*(1 - dr);
				float w01 = (1 - dr)*dc;
				float w10 = (1 - dc)*dr;
				float w11 = dr * dc;

				int bigOffset = numSheet * oldSheetCount;
				newDPtr[numSheet * newSheetCount + offset] =
					w00 * oldDPtr[bigOffset + backSheetOffset] +
					w01 * oldDPtr[bigOffset + ind_p01] +
					w10 * oldDPtr[bigOffset + ind_p10] +
					w11 * oldDPtr[bigOffset + ind_p11];
			}
			else{
				newDPtr[numSheet * newSheetCount + offset] = 0;
			}
		}
	}

	template <typename Dtype>
	void InterpImageNN_gpu(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped, const Interp& interp){
		//Get the paramters from the original and warped and apply the
		//transformation to it.
		const Dtype* orgDataPtr = orig->gpu_data();
		Dtype* warpedDataPtr = warped->mutable_gpu_data();
		int oldNPerSheet = orig->height() * orig->width();
		int newNPerSheet = warped->height() * warped->width();
		int nCount = warped->count();
		switch (interp)
		{
		case NN:
			nn_interpolation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, orgDataPtr, oldNPerSheet, 
				warpedDataPtr, newNPerSheet, coord);
			break;
		case BILINEAR:
			bilinear_interpolation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, orgDataPtr, oldNPerSheet, warpedDataPtr, 
				newNPerSheet, coord, orig->width());
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
		CUDA_POST_KERNEL_CHECK;
	}

	//explicit instantiation
	template void InterpImageNN_gpu(const Blob<float>* orig, const float* coord,
		Blob<float>* warped, const Interp& interp);
//	template void InterpImageNN_gpu(const Blob<double>* orig, const float* coord,
//		Blob<double>* warped, const Interp& interp);

	/*
	 *******PropagateErrorNN_gpu********
	 *If we kernalize each pixel in the top(warped image), bc of race conditions
	 *we need to use atomaticAdd, but it's slow and there is no double implementation
	 *of atomicAdd.
	 *So instead, parallelize over each pixel in the bottom (original) and for each pixel
	 * loop over the coord, find those top neurons that came from this bottom pixel and add.
	 * Similar to MaxPoolBackward Super. fucking slow. duh.
	 */
	template <typename Dtype>
	__global__ void PropagateErrorNN_kernel_nonatomic(
		const int nthreads, const Dtype* top_diff, const int width,
		const int height, const int channels, const int num,
		const int top_len, const float* coord, Dtype* bottom_diff){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			//find out the target index to look for in coord
			//can do this the way abhishek did so we can save on some computation(like
			//with SheetCount)
			int w = index % width;
			int h = (index / width) % height;
			int c = (index / width / height) % channels;
			int n = index / width / height / channels;

			int target_ind = h * width + w;
			//move over top_diff ptr to the beginning of its hxw sheet:
			//top_len = width_top * height_top
			top_diff += (n * channels + c) * top_len;

			Dtype gradient = 0;
			//loop over coord and add to grad if coord[i] == target_ind
			for (int i = 0; i < top_len; ++i){
				gradient += top_diff[i] * (static_cast<int>(coord[i]) == target_ind);
			}
			bottom_diff[index] += gradient;
		}
	}

	template <typename Dtype>
	__global__ void nn_backpropagation_kernel(int nThreads, const Dtype* topDataPtr,
		int topSheetCount, Dtype* bottomDataPtr,
		int bottomSheetCount, const float* coord){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nThreads){
			int offset = index % topSheetCount;
			int numSheet = index / topSheetCount;

			int bottomSheetOffset = static_cast<int>(coord[offset]);
			if (bottomSheetOffset >= 0){
				int bottomFinalOffset = numSheet * bottomSheetCount + bottomSheetOffset;
				//AJ: as atomicAdd is only available to float, this only works if
				//Dtype = float
				atomicAdd((&bottomDataPtr[bottomFinalOffset]),
					static_cast<float>(topDataPtr[numSheet * topSheetCount + offset]));
			}
		}
	}

	template <typename Dtype>
	__global__ void bilinear_backpropagation_kernel(int nThreads, const Dtype* topDataPtr,
		int topSheetCount, Dtype* bottomDataPtr, int bottomSheetCount,
		const float* coord, int W){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nThreads){
			int offset = index % topSheetCount;
			int numSheet = index / topSheetCount;
			int bottomSheetOffset = static_cast<int>(coord[offset]);
			if (bottomSheetOffset >= 0){
				int c0 = bottomSheetOffset % W;
				int ind_p11 = static_cast<int>(coord[offset + topSheetCount]);
				int c1 = ind_p11 % W;

				int ind_p01 = bottomSheetOffset - c0 + c1; //r0 * W + c1
				int ind_p10 = ind_p11 - c1 + c0;

				float dc = coord[offset + 2 * topSheetCount];
				float dr = coord[offset + 3 * topSheetCount];

				float w00 = (1 - dc)*(1 - dr);
				float w01 = (1 - dr)*dc;
				float w10 = (1 - dc)*dr;
				float w11 = dr * dc;

				float top_error = static_cast<float>(topDataPtr[index]);

				int commonOffset = numSheet * bottomSheetCount;

				//p00
				atomicAdd((&bottomDataPtr[commonOffset + bottomSheetOffset]),
					w00 * top_error);
				//p01
				atomicAdd((&bottomDataPtr[commonOffset + ind_p01]), w01 * top_error);
				//p10
				atomicAdd((&bottomDataPtr[commonOffset + ind_p10]), w10 * top_error);
				//p11
				atomicAdd(&bottomDataPtr[commonOffset + ind_p11], w11 * top_error);
			}
		}
	}

	template <typename Dtype>
	void BackPropagateErrorNN_gpu(const Blob<Dtype>* top, const float* coord,
		Blob<Dtype>* bottom, const Interp &interp){
	    //Get the parameters from the original and warped and apply the 
		//transformation to it.
		const Dtype* topDataPtr = top->gpu_diff();
		Dtype* bottomDataPtr = bottom->mutable_gpu_diff();
		int topNPerSheet = top->height() * top->width();
		int bottomNPerSheet = bottom->height() * bottom->width();
		//atomicAdd needs nTop many threads
		int nCount = top->count();
		switch (interp)
		{
		case NN:
			nn_backpropagation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, topDataPtr,
				topNPerSheet, bottomDataPtr, bottomNPerSheet, coord);
			break;
		case BILINEAR:
			bilinear_backpropagation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, topDataPtr,
				topNPerSheet, bottomDataPtr, bottomNPerSheet, coord, bottom->width());
			break;
		default:
			LOG(ERROR) << "Unknown interpolation mode " << interp;
			break;
		}
		CUDA_POST_KERNEL_CHECK;
	}

	//explicit instantiation
	template void BackPropagateErrorNN_gpu(const Blob<float>* top, const float* coord,
		Blob<float>* bottom, const Interp &interp);
//	template void PropagateErrorNN_gpu(const Blob<double>* top, const float* coord,
//		Blob<double>* bottom, const Interp &interp);

}