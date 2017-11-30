#ifndef CAFFE_IMP_MAP_LAYER_HPP_
#define CAFFE_IMP_MAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class ImpMapLayer : public Layer<Dtype> {
	public:
		explicit ImpMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "ImpMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		void create_hash_table();
		int height_, width_, channel_, num_;
		Dtype weight_;
		int ngroup_;
		int channel_out_;
		int max_channel_;
		int channels_per_group_;
		Dtype ratio_;
		Blob<Dtype> per_;
		Blob<Dtype> boundary_;
		Blob<int> hash_;
		Blob<int> imp_map_;
		Blob<Dtype> one_multiper_;
		bool global_gradient_;
		bool lquantize_;
	};
}

#endif  