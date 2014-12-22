#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void L1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
  }


  template <typename Dtype>
    void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      int num = bottom[0]->num();
      int count = bottom[0]->count();
      int dim = count / num;

      caffe_copy(count, bottom_data, bottom_diff);
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
          bottom_diff[i * dim + j] = std::abs(bottom_diff[i * dim + j]);
        }
      }
      Dtype* loss = top[0]->mutable_cpu_data();
      loss[0] = caffe_cpu_asum(count, bottom_diff) / num;

      return;
    }

  template <typename Dtype>
    void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int num = bottom[0]->num();
        int count = bottom[0]->count();
        int dim = count / num;

        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < dim; ++j) {
            bottom_diff[i * dim + j] = (bottom_data[i * dim + j] >= 0 ? 1 : -1);
          }
        }

        const Dtype loss_weight = top[0]->cpu_diff()[0];
        caffe_cpu_sign(count, bottom_diff, bottom_diff);
        caffe_scal(count, loss_weight / num, bottom_diff);
      }

      return;
    }

#ifdef CPU_ONLY
  STUB_GPU(L1LossLayer);
#endif
  INSTANTIATE_CLASS(L1LossLayer);
  REGISTER_LAYER_CLASS(L1_LOSS, L1LossLayer);
}  // namespace caffe
