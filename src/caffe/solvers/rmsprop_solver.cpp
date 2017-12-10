#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {


// gan added
template <typename Dtype>
void RMSPropSolver<Dtype>::RMSPropPreSolve() {
  // Add the extra history entries for RMSProp after those from
  // SGDSolver::PreSolve

  // history[0] if for D
  // history[1] if for G
  if (this->gan_solver_) {
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    for (int i = 0; i < net_params.size(); ++i) {
        const vector<int>& shape = net_params[i]->shape();
        this->history_.push_back(
                shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate);
#endif

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  Dtype rms_decay = this->param_.rms_decay();
	Dtype local_rate = rate * net_params_lr[param_id];
	//  current using history start point
	Blob<Dtype>* cur_param_history = this->history_[param_id].get();
  if (this->gan_solver_ && Net<Dtype>::get_gan_mode() == 1) {
      cur_param_history = this->history_[param_id + net_params.size()].get();
  }
  switch (Caffe::mode()) {
  case Caffe::CPU:
    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id] -> count(),
        Dtype(1-rms_decay), this->update_[param_id]->cpu_data(),
        rms_decay, cur_param_history->mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
        cur_param_history->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    caffe_add_scalar(net_params[param_id]->count(),
        delta, this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
        net_params[param_id]->cpu_diff(), this->update_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
        this->update_[param_id]->cpu_data(), Dtype(0),
        net_params[param_id]->mutable_cpu_diff());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    rmsprop_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        cur_param_history->mutable_gpu_data(),
        rms_decay, delta, local_rate);
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RMSPropSolver);
REGISTER_SOLVER_CLASS(RMSProp);

}  // namespace caffe
