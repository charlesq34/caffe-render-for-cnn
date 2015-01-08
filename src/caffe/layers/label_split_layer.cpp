#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LabelSplitLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
}

template <typename Dtype>
void LabelSplitLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype split_num = this->layer_param_.label_split_param().split_num();
  CHECK_EQ(top.size(), 2);
  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape(bottom[0]->num(), 1, 1, 1);
  }
}

template <typename Dtype>
void LabelSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // e.g. bottom_label \in [0,360), split_num = 4 => unit_len = 90
  // 30->(0,30); 124->(1, 34);
  const Dtype* bottom_label = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  Dtype period = this->layer_param_.label_split_param().period();
  Dtype split_num = this->layer_param_.label_split_param().split_num();
  Dtype label_max = this->layer_param_.label_split_param().label_max();
  float unit_len = period / float(split_num);

  for (int i = 0; i < num; ++i) {
    float label = bottom_label[i]/(label_max/period); // hardcoded, default label range: 0~360
    top[0]->mutable_cpu_data()[i] = floor(label / unit_len);
    top[1]->mutable_cpu_data()[i] = fmod(label, unit_len);
    //LOG(INFO) << "Label: " << bottom_label[i] << " Category: " << floor(bottom_label[i]/unit_len);
  }
}

INSTANTIATE_CLASS(LabelSplitLayer);
REGISTER_LAYER_CLASS(LABEL_SPLIT, LabelSplitLayer);
}  // namespace caffe

