//  Create on: 2016/10/21 ShanghaiTech
//  Author:    Yingying Zhang


#include <vector>
#include "caffe/layers/gan_gate_layer.hpp"
// gan added
#include "caffe/net.hpp"

namespace caffe {

  template <typename Dtype>
  void GANGateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    gan_mode_ = 1;
    top[0]->ReshapeLike(*bottom[0]);
  }

  template <typename Dtype>
  void GANGateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    gan_mode_ = Net<Dtype>::get_gan_mode();
    // LOG(INFO) << "Gate:  gan_mode"<<  gan_mode_ << std::endl;
    int index = gan_mode_ == 1 ? 0 : 1;
    // LOG(INFO) << "bottom[0]" << bottom[0]->asum_data();
    top[0]->ReshapeLike(*bottom[index]);
    top[0]->ShareData(*bottom[index]);
    top[0]->ShareDiff(*bottom[index]);
    
  }

  INSTANTIATE_CLASS(GANGateLayer);
  REGISTER_LAYER_CLASS(GANGate);

}  // namespace caffe