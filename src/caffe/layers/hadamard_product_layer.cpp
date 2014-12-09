#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HadamardProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  K_ = bottom[0]->count() / bottom[0]->num();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, K_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.hadamard_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void HadamardProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with hadamard product parameters.";
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void HadamardProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      top_data[i * dim + j] *= weight[j];
    }
  }

}

template <typename Dtype>
void HadamardProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype * weight_diff = this->blobs_[0]->mutable_cpu_diff()

    int num = bottom[0]->num(); // M_
    int count = bottom[0]->count();
    int dim = count / num; // K_

    for (int j = 0; j < dim; ++j) {
      weight_diff[j] = 0.0;
      for (int i = 0; i < num; ++i) {
	weight_diff[j] += (top_diff[i * dim + j] * bottom_data[i * dim + j]);
      }
    }
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff;

    int num = bottom[0]->num(); // M_
    int count = bottom[0]->count();
    int dim = count / num; // K_

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
	bottom_diff[i * dim + j] = (top_diff[i * dim + j] * weight[j]);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HadamardProductLayer);
#endif

INSTANTIATE_CLASS(HadamardProductLayer);
REGISTER_LAYER_CLASS(INNER_PRODUCT, HadamardProductLayer);
}  // namespace caffe
