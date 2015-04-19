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
void AccuracyViewLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  tol_angle_ = this->layer_param_.accuracy_view_param().tol_angle();
}

template <typename Dtype>
void AccuracyViewLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";

  // by rqi. HARD CODED -> number of classes == 4320
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), 4320); // note that bottom[1] (i.e. label) can be 0~4320, 4321 classes;

  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AccuracyViewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  int nonbkg_cnt = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num(); // should be 4320
  //vector<Dtype> maxval(top_k_+1);
  //vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    int label_angle = int(bottom_label[i]) % 360;
    int cls_idx = int(bottom_label[i]) / 360;
    if (cls_idx == 12) { continue; } // if it's bkg
    nonbkg_cnt++;
    
    int pred_angle = 0;
    Dtype max_prob = 0;
    for (int j = 0; j < 360; ++j) { // only consider probs within cls
      Dtype prob = bottom_data[i*dim + cls_idx*360 + j];
      if (prob > max_prob) {
        pred_angle = j;
        max_prob = prob;
      }
    }

    Dtype error = std::min(abs(pred_angle - label_angle), 360-abs(pred_angle - label_angle));
    if (error <= tol_angle_) {
      ++accuracy;
    } 
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / nonbkg_cnt;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyViewLayer);
REGISTER_LAYER_CLASS(ACCURACY_VIEW, AccuracyViewLayer);
}  // namespace caffe
