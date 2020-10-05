#ifndef CAFFE_CENTER_SIMILARITY_LAYER_HPP_
#define CAFFE_CENTER_SIMILARITY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A "similarity-measure" layer, computes an inner product
 *        with a set of video centers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CenterSimilarityLayer : public Layer<Dtype> {
 public:
  explicit CenterSimilarityLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterSimilarity"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }// frames fea, start end video index, video centers
  virtual inline int MinTopBlobs() const { return 1; }// similarity of frames, loss of video center align

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  unsigned int M_;// frame num
  unsigned int N_;// fea dim 
  unsigned int  video_num_;//input clips num, must identical for all batches
  Dtype loss_weight_;
  //Blob<Dtype> distance_;
  //Blob<Dtype> variation_sum_;
};

}  // namespace caffe

#endif  // CAFFE_CENTER_SIMILARITY_LAYER_HPP_
