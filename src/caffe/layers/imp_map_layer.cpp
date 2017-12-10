#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/imp_map_layer.hpp"
#include "thrust/host_vector.h"
#include "thrust/sort.h"
#include "thrust/execution_policy.h"
namespace caffe {
	template <typename Dtype>
	void ImpMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		ImpMapParameter rm = this->layer_param_.imp_map_param();
		//LOG(INFO) << "1L";
		weight_ = rm.weight();
		ratio_ = rm.cmp_ratio();
		ngroup_ = rm.groups();
		channel_out_ = rm.channel_out();
		max_channel_ = rm.max_channel();
		lquantize_ = rm.lquantize();
		global_gradient_ = (rm.method() == ImpMapParameter_ImpMethod_GLOBAL);
		CHECK_EQ(channel_out_%ngroup_, 0);
		//LOG(INFO) << "2L:"<<channel_out_<<","<<ngroup_;
		channels_per_group_ = channel_out_ / ngroup_;
		//sort_map_method_ = (rm.method() == ImpMapParameter_ImpMethod_SORT);
		per_.Reshape(1,1,1,ngroup_);
		boundary_.Reshape(1,1,1,ngroup_);
		int mod = ngroup_;
		hash_.Reshape(1, 1, 1, mod+1);
		one_multiper_.Reshape(1, 1, 1, channel_out_);
		//LOG(INFO) << "3L";
		caffe_set(channel_out_, Dtype(1.0), one_multiper_.mutable_cpu_data());
		//LOG(INFO) << "3.1L";
		// Dtype * per = per_.mutable_cpu_data();
		Dtype * bound = boundary_.mutable_cpu_data();
		/*if (sort_map_method_){
			for (int i = 0; i < ngroup_; i++)
			{
				per[i] = rm.ratio(i);
			}
		}else{}*/
		Dtype one = 1.0 / ngroup_;
		//LOG(INFO) << "3.2L"<<one;
		for (int i = 1; i <= ngroup_; i++)
		{
			bound[i - 1] = one * i;
		}
		create_hash_table();
		// const int * hash = hash_.cpu_data();
		// for (int i = 0; i <= mod; i++)
			// LOG(INFO) << hash[i];
		
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//LOG(INFO) << "4L";
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		channel_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		//LOG(INFO) << "5L";
		imp_map_.Reshape(num_, 1, height_, width_);
		top[0]->Reshape(num_,channel_out_,height_,width_);
		CHECK_EQ(channel_,1);
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::create_hash_table()
	{
		int mod =  ngroup_;
		Dtype one = 1.0 / mod;
		Dtype val;
		int * hash = hash_.mutable_cpu_data();
		const Dtype * bound = boundary_.cpu_data();
		int pr = 0;
		for (int i = 0; i < mod; i++)
		{
			val = one * i;
			if (val >= bound[pr])
				pr++;
			if (lquantize_){
				hash[i] = pr* channels_per_group_;
			}
			else{
				hash[i] = pr* channels_per_group_ + channels_per_group_;
			}
			
			//hash[i] = pr* channels_per_group_ + channels_per_group_;;
		}
		hash[mod] = channel_out_;
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		int * const imp_map = imp_map_.mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		unsigned int psize = num_*height_*width_;
		/*
		if (sort_map_method_){
			Dtype * const tmp = new Dtype[psize];
			const Dtype * per = per_.cpu_data(); //
			Dtype * bound = boundary_.mutable_cpu_data();
			Dtype pt=0;
			//LOG(INFO) << "1L";
			memcpy(tmp, bottom_data, sizeof(Dtype)*psize);
			//LOG(INFO) << "2L";
			thrust::sort(thrust::host, tmp, tmp + psize);
			for (int i = 0; i < ngroup_; i++)
			{
				
				pt = pt + per[i];
				int idx = int(pt*psize) - 1;
				if (idx < 0)
					idx = 0;
				bound[i] = tmp[idx];
			}
			create_hash_table();
		}*/
		const int * hash = hash_.cpu_data();
		int ptr = 0;
		int mod =  ngroup_;
		for (int i = 0; i < psize; i++)
		{
			ptr = int(bottom_data[i] * mod + 0.00001);
			imp_map[i] = hash[ptr];
		}
		for (int pn = 0; pn < num_; pn++)
		{
			for (int pc = 0; pc < channel_out_; pc++)
			{
				for (int ph = 0; ph < height_; ph++)
				{
					for (int pw = 0; pw < width_; pw++)
					{
						int idx = ((pn*channel_out_ + pc)*height_ + ph)*width_ + pw;
						if (pc >= imp_map[pn*height_*width_ + ph*width_ + pw])
							top_data[idx] = 0;
						else
							top_data[idx] = 1;
					}
				}
			}
		}
		Dtype sum_ratio = top[0]->asum_data();
		this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
	}
	template <typename Dtype>
	void implayer_backward_cpu_kernel(int idx, const Dtype * const top_diff,Dtype * const bottom_diff ,const int * const imp,int lbits,int group,int h, int w, int cout){
		bottom_diff[idx] = 0;
		int pn = idx / h / w;
		int pp = idx % (h*w);
		const Dtype * top = top_diff + pn*cout*h*w;
		int start_idx = imp[idx] - lbits >= 0 ? imp[idx] - lbits : 0;
		int end_idx = imp[idx] + lbits <= cout ? imp[idx] + lbits : cout;
		for (int i = start_idx; i < end_idx; i++){
			bottom_diff[idx] += top[i*h*w + pp];
		}
		bottom_diff[idx] = bottom_diff[idx] * group;
	}
	template <typename Dtype>
	void ImpMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// const Dtype * one_mult = one_multiper_.cpu_data();
		const Dtype * top_diff = top[0]->cpu_diff();
		const int * const imp_map = imp_map_.mutable_cpu_data();
		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
		/*
		for (int i = 0; i < num_; i++)
		{
			caffe_cpu_gemv<Dtype>(CblasTrans, channel_out_, width_*height_, 1.,
				top_diff, one_mult, 0.,bottom_diff);
			top_diff = top_diff + channel_out_*width_*height_;
			bottom_diff = bottom_diff + width_*height_;
		}*/
		for (int i = 0; i < num_*width_*height_; i++){
			implayer_backward_cpu_kernel(i, top_diff, bottom_diff, imp_map, channels_per_group_, ngroup_, height_, width_, channel_out_);
		}
		/*
		if (sort_map_method_){  
			Dtype scale = bottom[0]->asum_diff();
			scale = scale / bottom[0]->count();
			caffe_add_scalar(bottom[0]->count(), -scale, bottom[0]->mutable_cpu_diff());
		}
		else{
			
		}*/
		Dtype sum_ratio = top[0]->asum_data();
		//this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / (num_*width_*height_*channel_out_);
		if (sum_ratio>ratio_*num_*width_*height_*channel_out_)
		{
			caffe_add_scalar(bottom[0]->count(), weight_, bottom[0]->mutable_cpu_diff());
		}
		
	}

#ifdef CPU_ONLY
	STUB_GPU(ImpMapLayer);
#endif

	INSTANTIATE_CLASS(ImpMapLayer);
	REGISTER_LAYER_CLASS(ImpMap);

}  // namespace caffe
