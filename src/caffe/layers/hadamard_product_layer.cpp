#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void HadamardProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, bottom[0]->height(), bottom[0]->width()));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.hadamard_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void HadamardProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  CHECK_EQ(bottom[0]->height(), this->blobs_[0]->height())<< "Input size incompatible with hadamard product parameters.";
  CHECK_EQ(bottom[0]->width(), this->blobs_[0]->width()) << "Input size incompatible with hadamard product parameters.";
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
void HadamardProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int offset = top[0]->offset(n, c, h, w);
          int weight_diff_offset = this->blobs_[0]->offset(0, 0, h, w);
          top_data[offset] *= weight[weight_diff_offset];
        }
      }
    }
  }

  return;
}

template<typename Dtype>
void HadamardProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype * weight_diff = this->blobs_[0]->mutable_cpu_diff();

    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int weight_diff_offset = this->blobs_[0]->offset(0, 0, h, w);
        weight_diff[weight_diff_offset] = 0.0;
        for (int n = 0; n < num; ++n) {
          for (int c = 0; c < channels; ++c) {
            int offset = top[0]->offset(n, c, h, w);
            weight_diff[weight_diff_offset] += (top_diff[offset] * bottom_data[offset]);
          }
        }
      }
    }
  }

  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();

    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            int offset = top[0]->offset(n, c, h, w);
            int weight_diff_offset = this->blobs_[0]->offset(0, 0, h, w);
            bottom_diff[offset] = (top_diff[offset] * weight[weight_diff_offset]);
          }
        }
      }
    }
  }

  return;
}

#ifdef CPU_ONLY
STUB_GPU(HadamardProductLayer);
#endif

INSTANTIATE_CLASS(HadamardProductLayer);
REGISTER_LAYER_CLASS(HADAMARD_PRODUCT, HadamardProductLayer);
}  // namespace caffe
