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

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class HadamardProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  HadamardProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~HadamardProductLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HadamardProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(HadamardProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HadamardProductParameter* hadamard_product_param =
      layer_param.mutable_hadamard_product_param();
  shared_ptr<HadamardProductLayer<Dtype> > layer(
      new HadamardProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(HadamardProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    HadamardProductParameter* hadamard_product_param =
        layer_param.mutable_hadamard_product_param();
    hadamard_product_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<HadamardProductLayer<Dtype> > layer(
        new HadamardProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(HadamardProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    HadamardProductParameter* hadamard_product_param =
        layer_param.mutable_hadamard_product_param();
    hadamard_product_param->mutable_weight_filler()->set_type("gaussian");
    HadamardProductLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
