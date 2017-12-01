//  Create on: 2016/10/19 ShanghaiTech
//  Author:    Yingying Zhang

// #define EPS Dtype(1e-5)
#define EPS Dtype(0)
#include <algorithm>
#include <vector>

#include "caffe/layers/sigmoid_gan_loss_layer.hpp"
// gan added
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
inline Dtype max(const Dtype a, const Dtype b)
{
  return a>b?a:b;
}

template <typename Dtype>
void SigmoidGANDGLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      LossLayer<Dtype>::LayerSetUp(bottom, top);
      // diter_idx_ = 0;
      // giter_idx_ = 0;
      // dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
      // gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();
      gan_mode_ = 1;
}

template <typename Dtype>
void SigmoidGANDGLossLayer<Dtype>::Forward_cpu(
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
      loss -= std::log(max(EPS, sigmoid(score[i])));
    }
    
  }
  //when gan_mode_ = 2, the input of loss is D(G(z))
  //loss is discriminative loss: -log(1-D(G(z)))
  if (gan_mode_ == 2){
    for(int i = 0; i<batch_size; ++i) {
      loss -= std::log(max(EPS, 1 - sigmoid(score[i])));
    }
  }
  //when gan_mode_ = 3, the input of loss is D(G(z))
  //loss is generative loss: -log(D(G(z)))
  if (gan_mode_ == 3){
    // giter_idx_++;
    for(int i = 0; i<batch_size; ++i) {
      loss -= std::log(max(EPS, sigmoid(score[i])));
    }
  }
  
  
  loss /= static_cast<Dtype>(batch_size);
  // LOG(INFO) << "loss of gandglosslayer" << loss << std::endl;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SigmoidGANDGLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batch_size = bottom[0]->count();
  gan_mode_ = Net<Dtype>::get_gan_mode();
  //when gan_mode_ = 1, the input of loss is D(x)
  //backward for discriminative loss
  if (gan_mode_ == 1) {
    // if (diter_idx_ % dis_iter_ == 0 ) {
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = (bottom[0]->cpu_data()[i] - Dtype(1)) / static_cast<Dtype>(batch_size);
          // LOG(INFO) << "gan_mode(1) bottom[0]->diff[" << i << "]=" << bottom[0]->mutable_cpu_diff()[i] << std::endl;
          // LOG(INFO) << "gan_mode(1) bottom[0]->data[" << i << "]=" << bottom[0]->cpu_data()[i] << std::endl;
      }
    // } else {
    //   for (int i = 0; i<batch_size; ++i) {
    //     bottom[0]->mutable_cpu_diff()[i] = Dtype(0);
    //   }
    // }
  }
  //when gan_mode_ = 2, the input of loss is D(G(z))
  //backward for discriminative loss
  if (gan_mode_ == 2){
    // if (diter_idx_ % dis_iter_ == 0 ) {
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = bottom[0]->cpu_data()[i] / static_cast<Dtype>(batch_size);
      }
    // } else {
    //   for (int i = 0; i<batch_size; ++i) {
    //     bottom[0]->mutable_cpu_diff()[i] = Dtype(0);
    //   }
    // }
  }
  //when gan_mode_ = 3, the input of loss is D(G(z))
  //backward for generative loss
  if (gan_mode_ == 3){
    // if (giter_idx_ % gen_iter_ == 0 ) {
      for(int i = 0; i<batch_size; ++i) {
        bottom[0]->mutable_cpu_diff()[i] = (bottom[0]->cpu_data()[i] - Dtype(1)) / static_cast<Dtype>(batch_size);
      }
    // } else {
    //   for (int i = 0; i<batch_size; ++i) {
    //     bottom[0]->mutable_cpu_diff()[i] = Dtype(0);
    //   }
    // }
  }
}



INSTANTIATE_CLASS(SigmoidGANDGLossLayer);
REGISTER_LAYER_CLASS(SigmoidGANDGLoss);

}  // namespace caffe