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

// from PASCAL3D (0~11 + 12 as bkg) to PASCAL VOC (1~20 + 0 as bkg).
const int cls_idx_map[13] = {1,2,4,5,6,7,9,11,14,18,19,20,0};

template <typename Dtype>
void DetViewSplitLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
}

template <typename Dtype>
void DetViewSplitLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 2);
  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape(bottom[0]->num(), 1, 1, 1);
  }
}

template <typename Dtype>
void DetViewSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_label = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  /*
  // HARD CODED: label input: 0~4320, where 0~4319 for object, 4320 for bkg
  // or label input: 10000~14320, where 10000~14319 for object, 14320 for bkg
  // top[0] is for detection, top[1] is for view
  for (int i = 0; i < num; ++i) {
    Dtype label = bottom_label[i]; 
    top[1]->mutable_cpu_data()[i] = label;
    
    Dtype det_label = 0;
    if (label < 10000) {
      det_label = cls_idx_map[int(label) / 360];
    } else {
      det_label = cls_idx_map[int(label - 10000) / 360] + 10000;
    }
    top[0]->mutable_cpu_data()[i] = det_label;
    //LOG(INFO) << "Label: " << bottom_label[i] << " Category: " << floor(bottom_label[i]/unit_len);
  }
  */

  // HARD CODED: 
  // input: 
  //   label input: 0 for bkg, 1~4320 (correspond to original 0~4319)
  //   or label input (real image obj): 10000~14319
  // output: 
  //   1. det: {1,2,4,...,20} + 0 OR {10001, ...,10020} + 10000
  //   2. view: 0~4319 + 4320 OR 10000~14319 + 14320
  // top[0] is for detection, top[1] is for view
  for (int i = 0; i < num; ++i) {
    Dtype label = bottom_label[i];
    if (label == 0) { // bkg
      top[1]->mutable_cpu_data()[i] = 14320;
    } else if (label < 10000) { // render obj 1~4320 -> 0~4319 
      top[1]->mutable_cpu_data()[i] = label - 1;
    } else { // real obj 10000~14319
      top[1]->mutable_cpu_data()[i] = label;
    }
    
    Dtype det_label = 0;
    if (label == 0) {
      det_label = 0;
    } else if (label < 10000) {
      det_label = cls_idx_map[int(label - 1) / 360];
    } else {
      det_label = cls_idx_map[int(label - 10000) / 360] + 10000;
    }
    top[0]->mutable_cpu_data()[i] = det_label;
  }
}

INSTANTIATE_CLASS(DetViewSplitLayer);
REGISTER_LAYER_CLASS(DET_VIEW_SPLIT, DetViewSplitLayer);
}  // namespace caffe

