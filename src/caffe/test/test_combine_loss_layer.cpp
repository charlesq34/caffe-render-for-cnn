#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CombineLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CombineLossLayerTest()
      : blob_bottom_cat_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_bottom_data0_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_bottom_data1_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    for (int i = 0; i < blob_bottom_cat_->count(); ++i) {
      blob_bottom_cat_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_cat_);
    
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(180.0);
    UniformFiller<Dtype> label_filler(filler_param);
    label_filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    
    filler_param.set_min(0.0);
    filler_param.set_max(180.0);
    UniformFiller<Dtype> data_filler(filler_param);
    data_filler.Fill(this->blob_bottom_data0_);
    blob_bottom_vec_.push_back(blob_bottom_data0_);
    data_filler.Fill(this->blob_bottom_data1_);
    blob_bottom_vec_.push_back(blob_bottom_data1_);
    
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CombineLossLayerTest() {
    delete blob_bottom_cat_;
    delete blob_bottom_label_;
    delete blob_bottom_data0_;
    delete blob_bottom_data1_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_cat_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_data0_;
  Blob<Dtype>* const blob_bottom_data1_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CombineLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CombineLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  CombineLossParameter* combine_loss_param = layer_param.mutable_combine_loss_param();
  combine_loss_param->set_norm(CombineLossParameter_Norm_L1);
  CombineLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3);
}

TYPED_TEST(CombineLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  CombineLossParameter* combine_loss_param = layer_param.mutable_combine_loss_param();
  combine_loss_param->set_norm(CombineLossParameter_Norm_L2);
  CombineLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3);
}
}  // namespace caffe
