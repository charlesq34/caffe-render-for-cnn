#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SubtractionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //TODO:
}

template <typename Dtype>
void SubtractionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //TODO:
}

#ifdef CPU_ONLY
STUB_GPU(SubtractionLayer);
#endif

INSTANTIATE_CLASS(SubtractionLayer);
REGISTER_LAYER_CLASS(SUBTRACTION, SubtractionLayer);
}  // namespace caffe
