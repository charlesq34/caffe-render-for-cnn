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
void QuantAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);

  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void QuantAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void QuantAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  
  // Quantinization parameters
  // e.g. 0~360 quantinize to 16 categories {0,1,...,15}, 0~11.25 and 348.75~360 -> 0
  Dtype in_period = this->layer_param_.quant_accuracy_param().in_period();
  Dtype out_num = this->layer_param_.quant_accuracy_param().out_num();
  Dtype label_max = this->layer_param_.quant_accuracy_param().label_max();
  float width = in_period/float(out_num);
  
  for (int i = 0; i < num; ++i) {
    // pred and label are float and of [0, in_period]
    float pred = fmod(bottom_data[i] + width / 2.0, in_period);
    if (pred < 0) { pred += in_period; }
    float label = fmod(bottom_label[i]/float(label_max/in_period) + width / 2.0, in_period); //HACK
    if (label < 0) { label += in_period; }
    int pred_cat = floor(pred  / width);
    int label_cat = floor(label / width);
    LOG(INFO) << "Final Pred: " << bottom_data[i] << ", " << pred_cat << " Label: " << bottom_label[i]/float(label_max/in_period) << "," << label_cat;

    if (pred_cat == label_cat) {
      ++accuracy;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(QuantAccuracyLayer);
REGISTER_LAYER_CLASS(QUANT_ACCURACY, QuantAccuracyLayer);
}  // namespace caffe

