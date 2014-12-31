#include <vector>
#include <cmath>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CombineLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype split_num = this->layer_param_.combine_loss_param().split_num();
  CHECK_EQ(bottom.size(), 2+split_num);

  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[i]->channels(), 1);
    CHECK_EQ(bottom[i]->height(), 1);
    CHECK_EQ(bottom[i]->width(), 1);
  }
}

template <typename Dtype>
void CombineLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1, 1, 1, 1);
  diff_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void CombineLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int split_num = (int)this->layer_param_.mixed_pred_param().split_num();
  Dtype period = this->layer_param_.mixed_pred_param().period();
  Dtype unit_len = period / Dtype(split_num);

  const Dtype* bottom_cat = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype loss = 0.0;

  for (int i = 0; i < bottom[0]->num(); ++i) {
    int cat_idx = (int)bottom_cat[i];
    Dtype label = bottom_label[i]; // Note: label has already been converted to 0~180
    const Dtype* bottom_data = bottom[cat_idx+2]->cpu_data();
    Dtype pred = bottom_data[i];
    //LOG(INFO) << "Pred: " << pred << "\tLabel: " << label;
    loss += (pred - label) * (pred - label);
    diff_.mutable_cpu_data()[i] = (pred - label);
  }

  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / 2.0;
}

template <typename Dtype>
void CombineLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }

  int split_num = (int)this->layer_param_.mixed_pred_param().split_num();
  for (int i = 0; i < split_num; ++i) {
    if (propagate_down[i+2]) { 
      Dtype* bottom_diff = bottom[i+2]->mutable_cpu_diff();
      for (int j = 0; j < bottom[0]->num(); ++j) {
        if (bottom[0]->cpu_data()[j] == i) {
          bottom_diff[j] = diff_.cpu_data()[j] * top[0]->cpu_diff()[0] / bottom[0]->num(); 
        } else {
          bottom_diff[j] = 0;
        }
      }
    }
  }

}

INSTANTIATE_CLASS(CombineLossLayer);
REGISTER_LAYER_CLASS(COMBINE_LOSS, CombineLossLayer);
}  // namespace caffe
