#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define PI 3.14159265

namespace caffe {

template <typename Dtype>
void PeriodicLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);

  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  // x1 - x2
  diff_.Reshape(bottom[0]->num(), 1, 1, 1);
  
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }

}

template <typename Dtype>
void PeriodicLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  /*
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  */
  Dtype loss(0.0);
  Dtype period = this->layer_param_.periodic_loss_param().period();
  Dtype label_max = this->layer_param_.periodic_loss_param().label_max();
  for (int i=0; i<count; ++i) {
    diff_.mutable_cpu_data()[i] = bottom[0]->cpu_data()[i] - bottom[1]->cpu_data()[i]/float(label_max/period); // HACK 
  }
  for (int i = 0; i < bottom[0]->num(); ++i) {
    loss +=  (1 - cos(diff_.cpu_data()[i] / period * 2 * PI)) / Dtype(2);
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PeriodicLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype period = this->layer_param_.periodic_loss_param().period();
    for (int i = 0; i < bottom[0]->num(); ++i) {
      bottom_diff[i] =  top[0]->cpu_diff()[0] * sin(diff_.cpu_data()[i] / period * 2 * PI) * PI / bottom[0]->num() / period;
    }
  }
}

INSTANTIATE_CLASS(PeriodicLossLayer);
REGISTER_LAYER_CLASS(PERIODIC_LOSS, PeriodicLossLayer);
} // namespace caffe
