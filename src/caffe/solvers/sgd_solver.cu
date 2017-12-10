#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}

template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}

template void sgd_update_gpu<float>(int, float*, float*, float, float);
template void sgd_update_gpu<double>(int, double*, double*, double, double);

template <typename Dtype>
__global__ void ClipWeightsGPU(int N, Dtype* g, Dtype clip_weights) {
  CUDA_KERNEL_LOOP(i, N) {
    if (g[i] > clip_weights) {
      g[i] = clip_weights;
    } else if(g[i] < -clip_weights) {
      g[i] = -clip_weights;
    }
  }
}

template <typename Dtype>
void clip_weight_gpu(int N, Dtype* g, Dtype clip_weights) {
  ClipWeightsGPU<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, clip_weights);
  CUDA_POST_KERNEL_CHECK;
}

template void clip_weight_gpu<float>(int, float*, float);
template void clip_weight_gpu<double>(int, double*, double);
}  // namespace caffe
