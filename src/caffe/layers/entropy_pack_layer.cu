#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/entropy_pack_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void entropy_pack_kernel(const int nthreads, const Dtype* const bottom_data,
		const int ks, const int ch_out, const int channel,const int height, 
		const int width, Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int half_ks = ks / 2;
			int tw = index % ks;
			int th = (index / ks) % ks;
			int tc = (index / ks / ks) % ch_out;
			int tn = index / ks / ks / ch_out;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				//if ((tw > half_ks) || (th > half_ks)) continue;
				if (th > half_ks) continue;
			}
			int bw = tn % width;
			int bh = (tn / width) % height;
			int bc = (tn / width / height) % channel;
			int bn = tn / width / height / channel;
			int pw = bw + tw - half_ks;
			int ph = bh + th - half_ks;
			int pc = bc + tc - ch_out + 1;
			if (pw < 0 || pw >= width) continue;
			if (ph < 0 || ph >= height) continue;
			if (pc < 0) continue;
			int idx = ((bn*channel + pc)*height + ph)*width + pw;
			top_data[index] = bottom_data[idx]+1;
		}
	
	}
	template <typename Dtype>
	__global__ void entropy_pack_imp_kernel(const int nthreads, const Dtype* const bottom_data, 
		const Dtype* const imp_data,const int ks, const int ch_out, 
		const int channel, const int height,const int width, Dtype* const top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int half_ks = ks / 2;
			int tw = index % ks;
			int th = (index / ks) % ks;
			int tc = (index / ks / ks) % ch_out;
			int tn = index / ks / ks / ch_out;
			if (imp_data[tn]<0.5) continue;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				//if ((tw > half_ks) || (th > half_ks)) continue;
				if (th > half_ks) continue;
			}
			int bw = tn % width;
			int bh = (tn / width) % height;
			int bc = (tn / width / height) % channel;
			int bn = tn / width / height / channel;
			int pw = bw + tw - half_ks;
			int ph = bh + th - half_ks;
			int pc = bc + tc - ch_out + 1;
			if (pw < 0 || pw >= width) continue;
			if (ph < 0 || ph >= height) continue;
			if (pc < 0) continue;
			int idx = ((bn *channel + pc)*height + ph)*width + pw;
			if (imp_data[idx]>0)
				top_data[index] = bottom_data[idx]+1;
		}

	}
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* const top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* imp_data;
		int count = top[0]->count();
		start_idx = int(this->blobs_[0]->cpu_data()[0] + 0.001);
		//LOG(INFO) << start_idx << ":" << (start_idx + sample_num) % num_;
		this->blobs_[0]->mutable_cpu_data()[0] = Dtype((start_idx + sample_num) % num_);
		//LOG(INFO) << "rand:" << start_idx;
		bottom_data = bottom_data + start_idx*channel_*height_*width_;
		caffe_gpu_memcpy(top[1]->count()*sizeof(Dtype), bottom_data, top[1]->mutable_gpu_data());
		caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
		if (imp){
			imp_data = bottom[1]->gpu_data();
			imp_data = imp_data + start_idx*channel_*height_*width_;
			caffe_gpu_axpby(top[1]->count(), Dtype(1.0), imp_data, Dtype(1.0), top[1]->mutable_gpu_data());
			entropy_pack_imp_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, imp_data, kernel_size, channel_out_,
				 channel_, height_, width_, top_data);
		}
		else{
			caffe_gpu_add_scalar(top[1]->count(), Dtype(1.0), top[1]->mutable_gpu_data());
			entropy_pack_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_data, kernel_size, channel_out_,
				channel_, height_, width_, top_data);
		}
		CUDA_POST_KERNEL_CHECK;
	}
	template <typename Dtype>
	__global__ void entropy_pack_kernel_backward(const int nthreads, Dtype* const bottom_diff,
		const int ks, const int ch_out, const int channel, const int height,
		const int width, const Dtype* const top_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int half_ks = ks / 2;
			int tw = index % ks;
			int th = (index / ks) % ks;
			int tc = (index / ks / ks) % ch_out;
			int tn = index / ks / ks / ch_out;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				//if ((tw > half_ks) || (th > half_ks)) continue;
				if (th > half_ks) continue;
			}
			int bw = tn % width;
			int bh = (tn / width) % height;
			int bc = (tn / width / height) % channel;
			int bn = tn / width / height / channel;
			int pw = bw + tw - half_ks;
			int ph = bh + th - half_ks;
			int pc = bc + tc - ch_out + 1;
			if (pw < 0 || pw >= width) continue;
			if (ph < 0 || ph >= height) continue;
			if (pc < 0) continue;
			int idx = ((bn *channel + pc)*height + ph)*width + pw;
			//bottom_diff[idx] += top_diff[index];
			atomicAdd((float *)(bottom_diff + idx), (float)top_diff[index]);
		}

	}
	/*
	template <typename Dtype>
	__global__ void entropy_pack_imp_kernel_backward(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const imp_data, const int ks, const int ch_out,
		const int channel, const int height, const int width, const Dtype* const top_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int bw = index % width;
			int bh = (index / width) % height;
			int bc = (index / width / height) % channel;
			int bn = index / width / height / channel;
			int half_ks = ks / 2;
			int pc, ph, pw, pidx, tidx;
			if (imp_data[index] < 0.5) continue;
			for (int ct = 0; ct < ch_out - 1; ct++){
				for (int ch = 0; ch < ks; ch++){
					for (int cw = 0; cw < ks; cw++){
						pc = bc + (ch_out - 1 - ct);
						ph = bh + (half_ks - ch);
						pw = bw + (half_ks - cw);
						if (pc >= channel || pc < 0) continue;
						if (ph < 0 || ph >= height) continue;
						if (pw < 0 || pw >= width) continue;
						pidx = ((bn*channel + pc)*height + ph)*width + pw;
						if (imp_data[pidx] > 0){
							tidx = ((pidx*ch_out + ct)*ks + ch)*ks + cw;
							bottom_diff[index] += top_diff[tidx];
						}
					}
				}
			}
			for (int ch = 0; ch < half_ks; ch++){
				for (int cw = 0; cw < ks; cw++){
					ph = bh + (half_ks - ch);
					pw = bw + (half_ks - cw);
					if (ph < 0 || ph >= height) continue;
					if (pw < 0 || pw >= width) continue;
					pidx = ((bn*channel + bc)*height + ph)*width + pw;
					if (imp_data[pidx] > 0){
						tidx = ((pidx*ch_out + ch_out-1)*ks + ch)*ks + cw;
						bottom_diff[index] += top_diff[tidx];
					}
				}
			}
			for (int cw = 0; cw < half_ks; cw++){
					pw = bw + (half_ks - cw);
					if (pw < 0 || pw >= width) continue;
					pidx = ((bn*channel + bc)*height + bh)*width + pw;
					if (imp_data[pidx] > 0){
						tidx = ((pidx*ch_out + ch_out-1)*ks + half_ks)*ks + cw;
						bottom_diff[index] += top_diff[tidx];
					}
			}

		}
	}
	*/
	template <typename Dtype>
	__global__ void entropy_pack_imp_kernel_backward(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const imp_data, const int ks, const int ch_out, 
		const int channel, const int height, const int width, const Dtype* const top_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int half_ks = ks / 2;
			int tw = index % ks;
			int th = (index / ks) % ks;
			int tc = (index / ks / ks) % ch_out;
			int tn = index / ks / ks / ch_out;
			if (imp_data[tn]<0.5) continue;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				//if ((tw > half_ks) || (th > half_ks)) continue;
				if (th > half_ks) continue;
			}
			int bw = tn % width;
			int bh = (tn / width) % height;
			int bc = (tn / width / height) % channel;
			int bn = tn / width / height / channel;
			int pw = bw + tw - half_ks;
			int ph = bh + th - half_ks;
			int pc = bc + tc - ch_out + 1;
			if (pw < 0 || pw >= width) continue;
			if (ph < 0 || ph >= height) continue;
			if (pc < 0) continue;
			int idx = ((bn *channel + pc)*height + ph)*width + pw;
			if (imp_data[idx]>0)
				atomicAdd((float *)(bottom_diff+idx),(float) top_diff[index]);
				//bottom_diff[idx] += top_diff[index];
			
		}

	}
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * const top_diff = top[0]->gpu_diff();
		const Dtype * imp_data;
		Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
		int count = top[0]->count();
		bottom_diff = bottom_diff + start_idx*channel_*height_*width_;
		caffe_gpu_set(sample_num*channel_*width_*height_, Dtype(0), bottom_diff);
		if (imp){
			imp_data = bottom[1]->gpu_data() + start_idx*channel_*height_*width_;
			entropy_pack_imp_kernel_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_diff, imp_data, kernel_size, channel_out_,
				channel_, height_, width_, top_diff);
		}else{
			entropy_pack_kernel_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, bottom_diff, kernel_size, channel_out_, 
				channel_, height_, width_, top_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(EntropyPackLayer);

}  // namespace caffe
