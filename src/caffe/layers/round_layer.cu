#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/round_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void round_kernel(const int nthreads, const Dtype* const bottom_data,
		const int num, const int channels, const int height, const int width, Dtype* const top_data, bool flag) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// Flag 代表是普通ZERO_ONE_Round还是ONE_MINUS_ONE
			if (flag)
			{
				top_data[index] = bottom_data[index]>0.5 ? 1 : 0;
			}
			else
			{
				top_data[index] = bottom_data[index]>0 ? 1 : -1;
			}
		}
	}
	template <typename Dtype>
	__global__ void round_mul_kernel(const int nthreads, const Dtype* const bottom_data,
		const Dtype* const hash, Dtype* const top_data, int grp) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype tmp;
			if (bottom_data[index] < 0){
				tmp = 0;
			}
			else if (bottom_data[index] > 1){
				tmp = 1;
			}
			else{
				tmp = bottom_data[index];
			}
		    top_data[index] = hash[int(tmp*grp)];
		}
	}
	template <typename Dtype>
	void RoundLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		//const Dtype* const mod = model.gpu_data();
		int count = bottom[0]->count();
		if (mult){
			round_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, hash_.gpu_data(), top_data, groups);
		}
		else{
			round_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, num_, ch_, h_, w_, top_data, zero_one);
		}
		CUDA_POST_KERNEL_CHECK;
		//LOG(INFO) << "forward";
	}
	template <typename Dtype>
	__global__ void round_kernel_backward(const int nthreads,const Dtype* const top_diff, 
		const Dtype* const bottom_data, Dtype* const bottom_diff,bool flag) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			if (flag){
				if (bottom_data[index] >= 0 && bottom_data[index] <= 1)
					bottom_diff[index] = top_diff[index];
				else if (bottom_data[index]<0 && top_diff[index]<0)
						bottom_diff[index] = top_diff[index];
				else if (bottom_data[index]>1 && top_diff[index]>0)
						bottom_diff[index] = top_diff[index];
				else
						bottom_diff[index] = 0;
			}
			else{
				if (bottom_data[index] >= 0 && bottom_data[index] <= 1)
					bottom_diff[index] = top_diff[index];
				else if (bottom_data[index]<-1 && top_diff[index]<0)
						bottom_diff[index] = top_diff[index];
				else if (bottom_data[index]>1 && top_diff[index]>0)
						bottom_diff[index] = top_diff[index];
				else
						bottom_diff[index] = 0;
			}
		}
	}
	template <typename Dtype>
	void RoundLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* const top_diff = top[0]->gpu_diff();
		Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* const bottom_data = bottom[0]->gpu_data();
		int count = bottom[0]->count();
		round_kernel_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top_diff, bottom_data, bottom_diff, zero_one);
		//LOG(INFO) << "1";
		CUDA_POST_KERNEL_CHECK;
		Dtype sum_ratio;
		//LOG(INFO) << weight_;
		caffe_gpu_asum(top[0]->count(), top[0]->gpu_data(), &sum_ratio);
		this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / top[0]->count();
		if (sum_ratio>ratio_*top[0]->count())
		{
			caffe_gpu_add_scalar(bottom[0]->count(), scale, bottom[0]->mutable_gpu_diff());
		}
		//LOG(INFO) << "backward";
	}

	INSTANTIATE_LAYER_GPU_FUNCS(RoundLayer);

}  // namespace caffe
