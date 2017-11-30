//  Create on: 2016/10/19 ShanghaiTech
//  Author:    Yingying Zhang


#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/rand_vec_layer.hpp"

#define MAX_RANDOM 10000

namespace caffe {

  template <typename Dtype>
  Dtype RandVecLayer<Dtype>::GetRandom(const Dtype lower, const Dtype upper) {
    CHECK(data_rng_);
    CHECK_LT(lower, upper) << "Upper bound must be greater than lower bound!";
    caffe::rng_t* data_rng =
        static_cast<caffe::rng_t*>(data_rng_->generator());
    return static_cast<Dtype>(((*data_rng)()) % static_cast<unsigned int>(
        (upper - lower) * MAX_RANDOM)) / static_cast<Dtype>(MAX_RANDOM)+lower;
  }

  template <typename Dtype>
  void RandVecLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const RandVecParameter& rand_vec_param = this->layer_param_.rand_vec_param();
    batch_size_ = rand_vec_param.batch_size();
    channel_ = rand_vec_param.channel();
    height_ = rand_vec_param.height();
    width_ = rand_vec_param.width();
    lower_ = rand_vec_param.lower();
    upper_ = rand_vec_param.upper();
    repeat_ = rand_vec_param.repeat();
    iter_idx_ = 1;
    vector<int> top_shape(2);
    top_shape[0] = batch_size_;
    top_shape[1] = channel_;
    if (height_ >0 && width_>0) {
      top_shape.resize(4);
      top_shape[0] = batch_size_;
      top_shape[1] = channel_;
      top_shape[2] = height_;
      top_shape[3] = width_;
    }
    top[0]->Reshape(top_shape);
  }

  template <typename Dtype>
  void RandVecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if (iter_idx_ == 1) {
      const unsigned int data_rng_seed = caffe_rng_rand();
      data_rng_.reset(new Caffe::RNG(data_rng_seed));
      int count = top[0]->count();
      for (int i = 0; i<count; ++i) {
        top[0]->mutable_cpu_data()[i] = GetRandom(lower_, upper_);
      }
    }
    //update iter_idx_
    iter_idx_ = iter_idx_ == repeat_ ? 1 : iter_idx_ + 1;
  }

  INSTANTIATE_CLASS(RandVecLayer);
  REGISTER_LAYER_CLASS(RandVec);

}  // namespace caffe