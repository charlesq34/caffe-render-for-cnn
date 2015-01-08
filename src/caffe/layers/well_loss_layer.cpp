// Hao: this file follows from hinge_loss_layer.cpp
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
void WellLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  } 
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void WellLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  float slope = this->layer_param_.well_loss_param().slope();
  Dtype low = this->layer_param_.well_loss_param().low_bound();
  Dtype high = this->layer_param_.well_loss_param().high_bound();

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      Dtype * val = & bottom_diff[i * dim + j];
      if (*val >= low && *val <= high) {
	*val = 0;
      }
      else if (*val <= low) { 
	*val = -slope * (*val - low) * (*val - low) * (*val -low);
      }
      else if (*val >= high) {
	*val = slope * (*val - high) * (*val - high) * (*val - high);
      }
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void WellLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    float slope = this->layer_param_.well_loss_param().slope();
    Dtype low = this->layer_param_.well_loss_param().low_bound();
    Dtype high = this->layer_param_.well_loss_param().high_bound();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
	const Dtype * val = & bottom_data[i * dim + j];
	if (*val >= low && *val <= high) {
	  bottom_diff[i * dim + j] = 0;
	}
	else if (*val <= low) { 
	  bottom_diff[i * dim + j] *= (-3 * slope * (*val - low) * (*val - low));
	}
	else if (*val >= high) {
	  bottom_diff[i * dim + j] *= (3 * slope * (*val - high) * (*val - high));
	}
      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_cpu_sign(count, bottom_diff, bottom_diff);
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(WellLossLayer);
REGISTER_LAYER_CLASS(WELL_LOSS, WellLossLayer);
}  // namespace caffe

