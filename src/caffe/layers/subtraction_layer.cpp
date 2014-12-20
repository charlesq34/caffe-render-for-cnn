#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SubtractionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(this->layer_param().subtraction_param().has_mean_file(), true) << "Mean file required for SubtractionLayer";
  const string& mean_file = this->layer_param().subtraction_param().mean_file();
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  data_mean_.FromProto(blob_proto);
}

template <typename Dtype>
void SubtractionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mean_data = data_mean_.cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        top_data[top[0]->offset(0, c, h, w)] = bottom_data[bottom[0]->offset(0, c, h, w)] - mean_data[data_mean_.offset(0, c, h, w)];
      }
    }
  }
  return;
}

template <typename Dtype>
void SubtractionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SubtractionLayer);
#endif

INSTANTIATE_CLASS(SubtractionLayer);
REGISTER_LAYER_CLASS(SUBTRACTION, SubtractionLayer);
}  // namespace caffe
