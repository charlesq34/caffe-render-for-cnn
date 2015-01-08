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
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
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

  const Dtype* bottom_cat = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();

  for (int i = 0; i < num; ++i) {
    int idx = (int)bottom_cat[i]; // idx for data
    const Dtype* bottom_data = bottom[idx+2]->cpu_data();
    diff_.mutable_cpu_data()[i] =  (bottom_data[i] - bottom_label[i]);
  }
  
  Dtype* loss = top[0]->mutable_cpu_data();
  Dtype* diff_vec = diff_.mutable_cpu_data();
  switch (this->layer_param_.combine_loss_param().norm()) {
    case CombineLossParameter_Norm_L1:
      loss[0] = caffe_cpu_asum(count, diff_vec) / num;
      break;
    case CombineLossParameter_Norm_L2:
      loss[0] = caffe_cpu_dot(count, diff_vec, diff_vec) / num / Dtype(2);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
  }
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
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  int num = bottom[0]->num();
  int count = bottom[0]->count();
 
  for (int i = 0; i < split_num; ++i) {
    if (propagate_down[i+2]) { 
      Dtype* bottom_diff = bottom[i+2]->mutable_cpu_diff();
      for (int j = 0; j < bottom[0]->num(); ++j) {
        if (bottom[0]->cpu_data()[j] == i) { 
          switch (this->layer_param_.combine_loss_param().norm()) {
	    case CombineLossParameter_Norm_L1:
	      caffe_cpu_sign(1, &(diff_.cpu_data()[j]), &(bottom_diff[j]));
	      break;
	    case CombineLossParameter_Norm_L2:
	      bottom_diff[j] = diff_.cpu_data()[j];
	      break;
	    default:
	      LOG(FATAL) << "Unknown Norm";
	  }
          caffe_scal(1, loss_weight / num, &(bottom_diff[j]));
        } else {
          bottom_diff[j] = Dtype(0);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(CombineLossLayer);
REGISTER_LAYER_CLASS(COMBINE_LOSS, CombineLossLayer);
}  // namespace caffe
