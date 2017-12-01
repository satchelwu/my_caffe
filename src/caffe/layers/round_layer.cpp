#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/round_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void RoundLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		RoundParameter rm = this->layer_param_.round_param();
		has_constrain = rm.has_constrain();
		scale = rm.scale();
		ratio_ = rm.ratio();
		zero_one = (rm.method() == RoundParameter_RoundMethod_ZERO_ONE);
		mult = (rm.method() == RoundParameter_RoundMethod_MULTI);
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		this->blobs_[0]->mutable_cpu_data();   // it is important for syncedmem not initialized !!
		if (mult){
			groups = rm.groups();
			hash_.Reshape(1, 1, 1, groups + 1);
			Dtype * hash = hash_.mutable_cpu_data();
			Dtype pr = 1.0 / groups;
			for (int i = 0; i < groups; i++)
			{
				hash[i] = pr*i + pr;
			}
			hash[groups] = 1.0;
		}
	}
	template <typename Dtype>
	void RoundLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		//round_bit = 1<<(rm.round_bit()-1);
		//LOG(INFO) << round_bit;
		
		
		//LOG(INFO) << "init";
		
		/*
		Dtype * val = model.mutable_cpu_data();
		Dtype mod = Dtype(1.0) / round_bit;
		for (int i = 0; i <= round_bit; i++){
		val[i] = mod*i;
		}*/
	}
	template <typename Dtype>
	void RoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype * const top_data = top[0]->mutable_cpu_data();
		const Dtype * const bottom_data = bottom[0]->cpu_data();
		//const Dtype * const mod = model.cpu_data();
		//LOG(INFO) << round_bit;
		if (mult)
		{
			const Dtype * const hash = hash_.cpu_data();
			Dtype tmp;
			for (int i = 0; i < bottom[0]->count(); i++)
			{

				if (bottom_data[i] < 0){
					tmp = 0;
				}
				else if (bottom_data[i] > 1){
					tmp = 1;
				}
				else{
					tmp = bottom_data[i];
				}
				top_data[i] = hash[int(tmp * groups)];
			}
		}
		else{
			if (zero_one){
				for (int i = 0; i < bottom[0]->count(); i++)
				{

					top_data[i] = (bottom_data[i]>0.5) ? 1 : 0;
				}
			}
			else{
				for (int i = 0; i < bottom[0]->count(); i++)
				{

					top_data[i] = (bottom_data[i]>0) ? 1 : -1;
				}
			}
		}
		//LOG(INFO) << "forward";
	}

	template <typename Dtype>
	void RoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype * top_diff = top[0]->cpu_diff();
		const Dtype * bottom_data = bottom[0]->cpu_data();
		//caffe_set(bottom[0], Dtype(0), bottom_diff);
		if (zero_one){
			for (int i = 0; i < bottom[0]->count(); i++){
				if (bottom_data[i] >= 0 && bottom_data[i] <= 1)
					bottom_diff[i] = top_diff[i];
				else if (bottom_data[i] < 0 && top_diff[i] < 0)
					bottom_diff[i] = top_diff[i];
				else if (bottom_data[i] > 1 && top_diff[i] > 0)
					bottom_diff[i] = top_diff[i];
				else
					bottom_diff[i] = 0;
			}
		}
		else{
			for (int i = 0; i < bottom[0]->count(); i++){
				if (bottom_data[i] >= -1 && bottom_data[i] <= 1)
					bottom_diff[i] = top_diff[i];
				else if (bottom_data[i] < -1 && top_diff[i] < 0)
					bottom_diff[i] = top_diff[i];
				else if (bottom_data[i] > 1 && top_diff[i] > 0)
					bottom_diff[i] = top_diff[i];
				else
					bottom_diff[i] = 0;
			}
		}
		if (has_constrain){
			Dtype sum_ratio = top[0]->asum_data();
			this->blobs_[0]->mutable_cpu_data()[0] = sum_ratio / top[0]->count();
			if (sum_ratio>ratio_*top[0]->count())
			{
				caffe_add_scalar(bottom[0]->count(), scale, bottom[0]->mutable_cpu_diff());
			}
		}
		//LOG(INFO) << "backward";
	}

#ifdef CPU_ONLY
	STUB_GPU(RoundLayer);
#endif

	INSTANTIATE_CLASS(RoundLayer);
	REGISTER_LAYER_CLASS(Round);

}  // namespace caffe
