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
void TripleViewSplitLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
}

template <typename Dtype>
void TripleViewSplitLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 4);
  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape(bottom[0]->num(), 1, 1, 1);
  }
}

template <typename Dtype>
void TripleViewSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_label = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  // GOAL:
  // convert 12*360*180*180 number to class, azimuth, elevation and tilt.
  int divider1 = 360*180*180;
  int divider2 = 180*180;
  int divider3 = 180;

  for (int i = 0; i < num; ++i) {
    int label = int(bottom_label[i]);
    int class_idx = label / divider1;
    int tmp1 = label % divider1;
    int azimuth = tmp1 / divider2;
    int tmp2 = tmp1 % divider2;
    int elevation = tmp2 / divider3;
    int tilt = tmp2 % divider3;
    top[0]->mutable_cpu_data()[i] = Dtype(class_idx);
    top[1]->mutable_cpu_data()[i] = Dtype(azimuth * class_idx);
    top[2]->mutable_cpu_data()[i] = Dtype(elevation * class_idx);
    top[3]->mutable_cpu_data()[i] = Dtype(tilt * class_idx);
  }
}

INSTANTIATE_CLASS(TripleViewSplitLayer);
REGISTER_LAYER_CLASS(TRIPLE_VIEW_SPLIT, TripleViewSplitLayer);
}  // namespace caffe

