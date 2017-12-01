//  Create on: 2016/10/19 ShanghaiTech
//  Author:    Yingying Zhang

#ifndef CAFFE_SIGMOID_GAN_LOSS_LAYER_HPP_
#define CAFFE_SIGMOID_GAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/*This loss layer computes loss for GAN.
 *  1. loss for discriminator
 *     bottom[0] : D(x)
 *     bottom[1] : D(G(z))
 *  2. loss for generator
 *     bottom[0] : D(G(z))
 */

template <typename Dtype>
class SigmoidGANDGLossLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidGANDGLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param)  {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
      top[0]->Reshape(loss_shape);
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline const char* type() const { return "SigmoidGANDGLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//   int dis_iter_;
//   int gen_iter_;
  int gan_mode_;
//   int diter_idx_;
//   int giter_idx_;
};


}  // namespace caffe

#endif  // CAFFE_GAN_LOSS_LAYER_HPP_