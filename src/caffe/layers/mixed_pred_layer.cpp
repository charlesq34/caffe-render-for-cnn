#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MixedPredLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype split_num = this->layer_param_.mixed_pred_param().split_num();

  CHECK_EQ(bottom[0]->channels(), split_num); // softmax
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);

  CHECK_EQ(bottom.size(), split_num+1);

  for (int i = 1; i < split_num+1; ++i) { // inner product
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[i]->channels(), 1);
    CHECK_EQ(bottom[i]->height(), 1);
    CHECK_EQ(bottom[i]->width(), 1);
  }
}

template <typename Dtype>
void MixedPredLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void MixedPredLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  int split_num = (int)this->layer_param_.mixed_pred_param().split_num();
  Dtype period = this->layer_param_.mixed_pred_param().period();
  float unit_len = period / float(split_num);
  const Dtype* bottom_prob = bottom[0]->cpu_data();
  
  for (int n = 0; n < bottom[0]->num(); ++n) {
    float max_prob = 0;
    int max_prob_idx = 0;
    
    // TODO vectorize the code
    for (int i = 0; i < split_num; ++i) {
      if (bottom_prob[n*split_num + i] > max_prob) {
        max_prob = bottom_prob[n*split_num + i];
        max_prob_idx = i;
      }
    }

    const Dtype* pred_vec = bottom[max_prob_idx+1]->cpu_data();
    float pred = pred_vec[n];
    pred = std::max(0.0, (double)std::min(pred, unit_len)) + max_prob_idx * unit_len;
    top[0]->mutable_cpu_data()[n] = pred;
  }
}

INSTANTIATE_CLASS(MixedPredLayer);
REGISTER_LAYER_CLASS(MIXED_PRED, MixedPredLayer);
}  // namespace caffe
