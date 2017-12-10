#include <algorithm>
#include <vector>

#include "caffe/layers/wgan_loss_layer.hpp"
// gan added
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void WGANDGLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      LossLayer<Dtype>::LayerSetUp(bottom, top);
      gan_mode_ = 1;
}

// forward_cpu is true Wasserstein Distance
// backward_cpu is minimizing the negative WD
template <typename Dtype>
void WGANDGLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[0]->count();
  const Dtype* score = bottom[0]->cpu_data();
  Dtype loss(0.0);
  gan_mode_ = Net<Dtype>::get_gan_mode();
  //when gan_mode_ = 1, the input of loss is D(x)
  //loss is discriminative loss: -log(D(x))
  if (gan_mode_ == 1) {
    // diter_idx_++;
    for(int i = 0; i<batch_size; ++i) {
      loss += score[i];
    }
  }
  //when gan_mode_ = 2, the input of loss is D(G(z))
  //loss is discriminative loss: -log(1-D(G(z)))
  if (gan_mode_ == 2){
    for(int i = 0; i<batch_size; ++i) {
      loss -= score[i];
    }
  }
  //when gan_mode_ = 3, the input of loss is D(G(z))
  //loss is generative loss: -log(D(G(z)))
  if (gan_mode_ == 3){
    // giter_idx_++;
    for(int i = 0; i<batch_size; ++i) {
      loss -= score[i];
    }
  }
  loss /= static_cast<Dtype>(batch_size);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WGANDGLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batch_size = bottom[0]->count();
  //when gan_mode_ = 1, the input of loss is D(x)
  //backward for discriminative loss
  gan_mode_ = Net<Dtype>::get_gan_mode();
  if (gan_mode_ == 1) {
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = Dtype(-1) / static_cast<Dtype>(batch_size);
      }

  }
  //when gan_mode_ = 2, the input of loss is D(G(z))
  //backward for discriminative loss
  if (gan_mode_ == 2){
 
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = Dtype(1) / static_cast<Dtype>(batch_size);
      }
  }
  //when gan_mode_ = 3, the input of loss is D(G(z))
  //backward for generative loss
  if (gan_mode_ == 3){
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = Dtype(-1) / static_cast<Dtype>(batch_size);
      }
  }
}



INSTANTIATE_CLASS(WGANDGLossLayer);
REGISTER_LAYER_CLASS(WGANDGLoss);

}  // namespace caffe