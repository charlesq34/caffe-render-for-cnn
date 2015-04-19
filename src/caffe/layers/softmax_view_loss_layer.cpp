#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithViewLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // by rqi
  weights_sum_ = 0.0;
  Dtype bandwidth = this->layer_param_.softmax_with_view_loss_param().bandwidth();
  CHECK_GT(180, bandwidth);
  CHECK_GE(bandwidth, 0);
  Dtype sigma = this->layer_param_.softmax_with_view_loss_param().sigma();
  CHECK_GT(sigma, 0);
  for (int k = -1*bandwidth; k<=bandwidth; k++) {
    weights_sum_ += exp(-abs(k)/float(sigma));
  }
}


template <typename Dtype>
void SoftmaxWithViewLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithViewLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //const Dtype* prob_data = prob_.cpu_data();
  Dtype* prob_data = prob_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();

  // added by rqi
  CHECK_EQ(spatial_dim, 1);
  Dtype bandwidth = this->layer_param_.softmax_with_view_loss_param().bandwidth();
  Dtype sigma = this->layer_param_.softmax_with_view_loss_param().sigma();
  // load loss weight, pos->render, neg->real
  Dtype pos_weight = this->layer_param_.softmax_with_view_loss_param().pos_weight();
  Dtype neg_weight = this->layer_param_.softmax_with_view_loss_param().neg_weight();
  
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

      //std::cout << dim << " " << label_value << " " << spatial_dim << "\n";
      CHECK_GT(dim, label_value * spatial_dim);
      
      // Added by rqi, full of HARD CODING..
      // ASSUMPTION: classes number is 12*360 or 12*360 + 1
      int cls_idx = label_value / 360; // 0~11,12->bkg
      if (cls_idx == 12) { continue; } // no loss for bkg

      // normalize prob_data of sample i to probs (360 numbers) 
      // inside the category corresponding to label_value
      Dtype probs_cls_sum = caffe_cpu_asum(360, &(prob_data[i*dim + cls_idx*360]));
      caffe_scal(360, Dtype(1.0/probs_cls_sum), &(prob_data[i*dim + cls_idx*360]));

      // convert to 360-class label
      int view_label = label_value % 360;
      Dtype tmp_loss = 0;
      for (int k = -1*bandwidth; k<=bandwidth; k++) {
          // get positive modulo
          // e.g. view_label+k=-3 --> 357
          int view_k = ((view_label + k) % 360 + 360) % 360;
          // convert back to 4320-class label
          int label_value_k = view_k + cls_idx * 360;
          // loss is weighted by exp(-|dist|/sigma)
          tmp_loss -= exp(-abs(k)/float(sigma)) * log(std::max(prob_data[i * dim +
          label_value_k * spatial_dim + j],Dtype(FLT_MIN)));
      }
      // scale loss with loss weight
      loss += tmp_loss * weight;
      
      //loss -= log(std::max(prob_data[i * dim +
      //    label_value * spatial_dim + j],Dtype(FLT_MIN)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithViewLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype bandwidth = this->layer_param_.softmax_with_view_loss_param().bandwidth();
    Dtype sigma = this->layer_param_.softmax_with_view_loss_param().sigma();
    // load loss weight, pos->render, neg->real
    Dtype pos_weight = this->layer_param_.softmax_with_view_loss_param().pos_weight();
    Dtype neg_weight = this->layer_param_.softmax_with_view_loss_param().neg_weight();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    // set bottom_diff to 0 at first
    caffe_set(prob_.count(), Dtype(0.0), bottom_diff);
    //caffe_copy(prob_.count(), prob_data, bottom_diff);

    Dtype weight = 0;
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

         // Added by rqi, full of HARD CODING..
         // ASSUMPTION: classes number is 12*360 or 12*360 + 1
         int cls_idx = label_value / 360; // 0~11,12->bkg

         if (cls_idx == 12) { // no gradient for bkg
           //caffe_set(dim, Dtype(0.0), &(bottom_diff[i*dim]));
           continue; 
         }
      
         // copy newly normalized probs of cls_idx to bottom_diff
         caffe_copy(360, &(prob_data[i*dim + cls_idx*360]), &(bottom_diff[i*dim + cls_idx*360]));
         // by rqi, scale by sum of weights
         caffe_scal(360, weights_sum_, &(bottom_diff[i*dim + cls_idx*360]));
         
         // scale prob_data part in bottom_diff
         caffe_scal(dim, weight, &(bottom_diff[i*dim]));
 
         // convert to 360-class label
         int view_label = label_value % 360;
         for (int k = -1*bandwidth; k<=bandwidth; k++) {
             // get positive modulo
             // e.g. view_label+k=-3 --> 357
             int view_k = ((view_label + k) % 360 + 360) % 360;
             // convert back to 4320-class label
             int label_value_k = view_k + cls_idx * 360;
             // loss is weighted by exp(-|dist|/sigma)
             // note: scale diff with loss weight
             bottom_diff[i * dim + label_value_k * spatial_dim + j] -= exp(-abs(k)/float(sigma)) * weight;
         }

         //bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
         //  * spatial_dim + j] -= 1;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftmaxWithViewLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_WITH_VIEW_LOSS, SoftmaxWithViewLossLayer);
}  // namespace caffe
