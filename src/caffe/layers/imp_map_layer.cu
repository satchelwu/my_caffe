#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/imp_map_layer.hpp"
#include "thrust/device_vector.h"
#include "thrust/copy.h"
#include "thrust/sort.h"
//#include "thrust/host_vector.h"
namespace caffe {


	template <typename Dtype>
	__global__ void imp_copy_kernel(const int nthreads, const Dtype* const src_data,
		Dtype* const dst_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			dst_data[index] = src_data[index];
		}
	}
	template <typename Dtype>
	__global__ void imp_forward_kernel(const int nthreads, const Dtype* const bottom_data,
		Dtype* const top_data,const int* const hash,int * const imp,const int num,const int channel_out, 
		const int base_space, const int mod,const int max_channel) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int n = index / base_space;
			int pxy = index % base_space;
			int out_space = channel_out*base_space;
			int ch = hash[int(bottom_data[index] * mod+0.00001)];
			//if (bottom_data[index] * mod < 0.01)
			//	ch = 0;
			//ch = ch < max_channel ? ch : max_channel;
			imp[index] = ch;
			for (int i = 0; i < ch; i++)
			{
				top_data[ n*out_space+ i*base_space+ pxy]=1.0;
			}
		}
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int cnt = bottom[0]->count();
		caffe_gpu_set<Dtype>(top[0]->count(), 0, top[0]->mutable_gpu_data());
		const Dtype * bottom_data = bottom[0]->gpu_data();
		/*
		if (sort_map_method_){
			thrust::device_vector<Dtype> dt_a(cnt);
			Dtype * raw_ptr = thrust::raw_pointer_cast(dt_a.data());
			imp_copy_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
				(cnt, bottom_data, raw_ptr);
			CUDA_POST_KERNEL_CHECK;
			thrust::sort(dt_a.begin(), dt_a.end());
			const Dtype * per = per_.cpu_data(); //
			Dtype * bound = boundary_.mutable_cpu_data();
			Dtype pt = 0;
			for (int i = 0; i < ngroup_; i++)
			{
				pt = pt + per[i];
				int idx = int(pt*cnt)-1;
				if (idx < 0)
					idx = 0;
				bound[i] = dt_a[idx];
			}
			LOG(INFO) << "3L";
			create_hash_table();
		}*/
		int mod =  ngroup_;
		//const int nthreads, const Dtype* const bottom_data,
		//	Dtype* const top_data, const Dtype* const hash, const int num, const int channel_out,
		//	const int base_space, const int mod
		imp_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
			(cnt, bottom_data,top[0]->mutable_gpu_data(),hash_.gpu_data(),imp_map_.mutable_gpu_data(),num_,channel_out_,width_*height_, mod,max_channel_);
		CUDA_POST_KERNEL_CHECK;
		Dtype sum_ratio;
		caffe_gpu_asum(top[0]->count(), top[0]->gpu_data(), &sum_ratio);
		this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
	}
	template <typename Dtype>
	__global__ void imp_backward_kernel_global(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const top_diff,  const int channel_out,const int base_space) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int n = index / base_space;
			int pxy = index % base_space;
			int out_space = channel_out*base_space;
			bottom_diff[index] = 0;
			for (int i = 0; i < channel_out; i++)
			{
				bottom_diff[index]+=top_diff[n*out_space + i*base_space + pxy];
			}
			//bottom_diff[index] = bottom_diff[index] * groups;
		}
	}
	template <typename Dtype>
	__global__ void imp_backward_kernel_local(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const top_diff, const int * const imp, const int lbits, const int groups, const int channel_out, const int base_space) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int n = index / base_space;
			int pxy = index % base_space;
			int out_space = channel_out*base_space;
			int start_idx = imp[index] - lbits >= 0 ? imp[index] - lbits : 0;
			int end_idx = imp[index] + lbits <= channel_out ? imp[index] + lbits : channel_out;
			bottom_diff[index] = 0;
			for (int i = start_idx; i < end_idx; i++)
			{
				bottom_diff[index] += top_diff[n*out_space + i*base_space + pxy];
			}
			bottom_diff[index] = bottom_diff[index] * groups;
		}
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype * bottom_data = bottom[0]->gpu_data();
		const int cnt = bottom[0]->count();
		caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
		if (global_gradient_)
		{
			imp_backward_kernel_global<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
				(cnt, bottom_diff, top[0]->gpu_diff(), channel_out_, width_*height_);
		}
		else{
			imp_backward_kernel_local<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
				(cnt, bottom_diff, top[0]->gpu_diff(), imp_map_.gpu_data(), channels_per_group_, ngroup_, channel_out_, width_*height_);
		}
		//imp_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS >> >
		//	(cnt, bottom_diff, top[0]->gpu_diff(), imp_map_.gpu_data(), channels_per_group_, ngroup_, channel_out_, width_*height_);
		//LOG(INFO) << "1 L: " << channels_per_group_<<" "<<ngroup_;
		CUDA_POST_KERNEL_CHECK;
		/*
		if (sort_map_method_){
			Dtype scale;
			caffe_gpu_asum(bottom[0]->count(), bottom[0]->gpu_diff(), &scale);
			scale = scale / bottom[0]->count();
			caffe_gpu_add_scalar(bottom[0]->count(), -scale, bottom[0]->mutable_gpu_diff());
		}
		else{}*/
		Dtype sum_ratio;
		//LOG(INFO) << weight_;
		
		caffe_gpu_asum(top[0]->count(), top[0]->gpu_data(), &sum_ratio);
		//LOG(INFO) << sum_ratio << " " << ratio_*num_*width_*height_*channel_out_;
		if (sum_ratio>ratio_*num_*width_*height_*channel_out_)
		{
			//LOG(INFO) << "CONS";
			caffe_gpu_add_scalar(bottom[0]->count(), weight_, bottom[0]->mutable_gpu_diff());
		}
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ImpMapLayer);

}  // namespace caffe
