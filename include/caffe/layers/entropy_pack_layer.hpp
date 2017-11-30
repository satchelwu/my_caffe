#ifndef CAFFE_ENTROPY_PACK_LAYER_HPP_
#define CAFFE_ENTROPY_PACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


// message EntropyPackParameter{
// 	optional uint32 kernel = 1[default=5];
// 	optional uint32 channels = 2 [default=8];
// 	optional uint32 samples = 3 [default=4];
// }


namespace caffe {

	template <typename Dtype>
	class EntropyPackLayer : public Layer<Dtype> {
	public:
		explicit EntropyPackLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "EntropyPack"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 2; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int kernel_size,channel_size;
		int start_idx, sample_num;
		int num_, channel_, width_, height_;
		int num_out_, channel_out_, width_out_, height_out_;
		bool imp;
	};
}

#endif  