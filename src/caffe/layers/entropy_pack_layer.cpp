#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/entropy_pack_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		srand(time(NULL));
		EntropyPackParameter rm = this->layer_param_.entropy_pack_param();
		kernel_size = rm.kernel();
		channel_size = rm.channels();
		sample_num = rm.samples();
		imp = (bottom.size() == 2);
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		this->blobs_[0]->mutable_cpu_data()[0] = 0;
	}
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		CHECK_EQ(num_%sample_num, 0);
		//this->blobs_[0]->mutable_cpu_data()[0] = 0;
		//EntropyPackParameter rm = this->layer_param_.entropy_pack_param();
		//kernel_size = rm.kernel();
		//channel_size = rm.channels();
		//sample_num = rm.samples();
		num_out_ = sample_num*channel_*width_*height_;
		channel_out_ = channel_size;
		width_out_ = kernel_size;
		height_out_ = kernel_size;
		top[0]->Reshape(num_out_,channel_out_,width_out_,height_out_);
		top[1]->Reshape(num_out_, 1, 1, 1);
		// 当前是否为编码importance map
		imp = (bottom.size() == 2);
	}


	template <typename Dtype>
	void entropy_pack_kernel_cpu(const int nthreads, const Dtype* const bottom_data,
		const int ks, const int ch_out,  const int channel, const int height,
		const int width, Dtype* const top_data) {
		for (int idx = 0; idx < nthreads;idx++) {
			int half_ks = ks / 2;
			int tw = idx % ks;
			int th = (idx / ks) % ks;
			int tc = (idx / ks / ks) % ch_out;
			int tn = idx / ks / ks / ch_out;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				if ((th > half_ks)) continue;
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
			int tidx = ((bn *channel + pc)*height + ph)*width + pw;
			top_data[idx] = bottom_data[tidx]+1;
		}

	}
	template <typename Dtype>
	void entropy_pack_imp_kernel_cpu(const int nthreads, const Dtype* const bottom_data,
		const Dtype* const imp_data, const int ks, const int ch_out,
		const int channel, const int height, const int width, Dtype* const top_data) {
		for (int idx = 0; idx < nthreads;idx++) {
			int half_ks = ks / 2;
			int tw = idx % ks;
			int th = (idx / ks) % ks;
			int tc = (idx / ks / ks) % ch_out;
			int tn = idx / ks / ks / ch_out;
			if (imp_data[tn]<0.5) continue;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				if ((th > half_ks)) continue;
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
			int tidx = ((bn*channel + pc)*height + ph)*width + pw;
			if (imp_data[tidx]>0)
				top_data[idx] = bottom_data[tidx]+1;
		}

	}
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * top_data = top[0]->mutable_cpu_data();
		const Dtype * imp_data;
		const Dtype * bottom_data = bottom[0]->cpu_data();
		// const Dtype * tmp;
		// const Dtype * imp_tmp;
		int count = top[0]->count();
		// Start From 0
		start_idx = int(this->blobs_[0]->cpu_data()[0]+0.001);
		// Sample_num 采样间隔
		this->blobs_[0]->mutable_cpu_data()[0] = Dtype((start_idx + sample_num) % num_);
		bottom_data = bottom_data + start_idx*channel_*height_*width_;
		caffe_copy(top[1]->count(), bottom_data, top[1]->mutable_cpu_data());  // elabel
		//LOG(INFO) << "start_idx:" << start_idx;
		//LOG(INFO) << bottom_data[0] << " " << bottom_data[1] << " " << bottom_data[2];
		caffe_set(top[0]->count(),Dtype(0),top_data);
		if (imp){
			imp_data = bottom[1]->cpu_data();
			imp_data = imp_data + start_idx*channel_*height_*width_;
			// top[1] = top[1] + imp_data
			caffe_cpu_axpby(top[1]->count(), Dtype(1.0), imp_data, Dtype(1.0), top[1]->mutable_cpu_data());
			entropy_pack_imp_kernel_cpu<Dtype>(count, bottom_data, imp_data, kernel_size, channel_out_,
				channel_, height_, width_, top_data);
		}else{
			caffe_add_scalar(top[1]->count(), Dtype(1.0), top[1]->mutable_cpu_data());
			entropy_pack_kernel_cpu<Dtype> (count, bottom_data, kernel_size, channel_out_,
				channel_, height_, width_, top_data);
		}
	}
	template <typename Dtype>
	void entropy_pack_kernel_backward_cpu(const int nthreads, Dtype* const bottom_diff,
		const int ks, const int ch_out, const int channel, const int height,
		const int width, const Dtype* const top_diff) {
		for (int idx=0; idx < nthreads;idx++) {
			int half_ks = ks / 2;
			int tw = idx % ks;
			int th = (idx / ks) % ks;
			int tc = (idx / ks / ks) % ch_out;
			int tn = idx / ks / ks / ch_out;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				if ((th > half_ks)) continue;
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
			int tidx = ((bn *channel + pc)*height + ph)*width + pw;
			bottom_diff[tidx] += top_diff[idx];
		}

	}
	template <typename Dtype>
	void entropy_pack_imp_kernel_backward_cpu(const int nthreads, Dtype* const bottom_diff,
		const Dtype* const imp_data, const int ks, const int ch_out,
		const int channel, const int height, const int width, const Dtype* const top_diff) {
		for (int idx=0; idx < nthreads;idx++) {
			int half_ks = ks / 2;
			int tw = idx % ks;
			int th = (idx / ks) % ks;
			int tc = (idx / ks / ks) % ch_out;
			int tn = idx / ks / ks / ch_out;
			if (imp_data[tn]<0.5) continue;
			if (tc == ch_out - 1){
				if ((tw >= half_ks) && (th == half_ks)) continue;
				if ((th > half_ks)) continue;
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
			int tidx = ((bn *channel + pc)*height + ph)*width + pw;
			if (imp_data[tidx]>0)
				bottom_diff[tidx] += top_diff[idx];
		}

	}
	template <typename Dtype>
	void EntropyPackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype * top_diff = top[0]->cpu_diff();
		const Dtype * imp_data;
		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
		int count = top[0]->count();
		
		bottom_diff = bottom_diff + start_idx*channel_*height_*width_;
		caffe_set(sample_num*channel_*width_*height_, Dtype(0), bottom_diff);
		if (imp){
			imp_data = bottom[1]->cpu_data() + start_idx*channel_*height_*width_;
			entropy_pack_imp_kernel_backward_cpu<Dtype>(count, bottom_diff, imp_data, kernel_size, channel_out_,
				 channel_, height_, width_, top_diff);
		}else{
			entropy_pack_kernel_backward_cpu<Dtype>(count, bottom_diff, kernel_size, channel_out_,
				 channel_, height_, width_, top_diff);
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(EntropyPack);
#endif

	INSTANTIATE_CLASS(EntropyPackLayer);
	REGISTER_LAYER_CLASS(EntropyPack);

}  // namespace caffe
