#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxPosNegWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxPosNegWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxPosNegWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();

  // by rqi
  // load loss weight, pos->render, neg->real
  Dtype pos_weight = this->layer_param_.softmax_pos_neg_loss_param().pos_weight();
  Dtype neg_weight = this->layer_param_.softmax_pos_neg_loss_param().neg_weight();

  Dtype weight = 0;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int label_value = static_cast<int>(label[i * spatial_dim + j]);
      
      // convert label_value to positive, mark loss weight
      if (label_value < 10000) {
        weight = pos_weight; 
      } else {
        weight = neg_weight;
        label_value = label_value - 10000;
      }
      
      CHECK_GT(dim, label_value * spatial_dim);
      
      // scale loss with loss weight 
      loss -= log(std::max(prob_data[i * dim +
          label_value * spatial_dim + j],Dtype(FLT_MIN))) * weight;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxPosNegWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // by rqi
    // load loss weight, pos->render, neg->real
    Dtype pos_weight = this->layer_param_.softmax_pos_neg_loss_param().pos_weight();
    Dtype neg_weight = this->layer_param_.softmax_pos_neg_loss_param().neg_weight();
    Dtype weight = 0;

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        int label_value = static_cast<int>(label[i * spatial_dim + j]);
      
        // convert label_value to positive, mark loss weight
        if (label_value < 10000) {
          weight = pos_weight; 
        } else {
          weight = neg_weight;
          label_value = label_value - 10000;
        }

        // scale prob_data part in bottom_diff
        caffe_scal(dim, weight, &(bottom_diff[i*dim]));
        
        // scale diff with loss weight
        bottom_diff[i * dim + label_value * spatial_dim + j] -= 1 * weight;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxPosNegWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxPosNegWithLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_POS_NEG_LOSS, SoftmaxPosNegWithLossLayer);
}  // namespace caffe
