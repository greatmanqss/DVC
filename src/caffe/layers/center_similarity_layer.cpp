#include <vector>

#include "caffe/layers/center_similarity_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterSimilarityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 2);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count()/bottom[0]->num(), bottom[2]->count()/bottom[2]->num());
  video_num_ = bottom[1]->num();
  M_ = bottom[0]->num();
  N_ = bottom[0]->count()/bottom[0]->num();
  loss_weight_ = 0.1;//qss
}

template <typename Dtype>
void CenterSimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(M_,1,1,1);
  //distance_.ReshapeLike(*bottom[0]);
  if(top.size()>1){
	  top[1]->Reshape(1,1,1,1);
  }// video center align loss 
}

template <typename Dtype>
void CenterSimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* seidx_data = bottom[1]->cpu_data();
  const Dtype* center_data = bottom[2]->cpu_data();

  Dtype* top_data = top[0]->mutable_cpu_data();
  //caffe_set(M_, Dtype(0.), top_data);
  int count = 0;
  for(int clip_id = 0; clip_id < video_num_; clip_id++)
  {
	int start = seidx_data[clip_id*2];
	int end = seidx_data[clip_id*2+1];
	count += (end-start+1);
	for (int frame_id = start; frame_id <= end; frame_id++)
	{
		Dtype dot = caffe_cpu_dot(N_, bottom_data + frame_id * N_, center_data + clip_id * N_);
		top_data[frame_id] = (dot + 1.0)/2.0;
	}
  }
  M_ = count; 
  if(top.size()>1){
	Dtype loss = caffe_cpu_asum(M_, top_data);
	top[1]->mutable_cpu_data()[0] = -loss_weight_ * loss/M_;
  }   
}

template <typename Dtype>
void CenterSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
		
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* seidx_data = bottom[1]->cpu_data();
  const Dtype* center_data = bottom[2]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  // back propagate to bottom 
  if (propagate_down[0]) {
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
    // Gradient with respect to bottom data
	for(int clip_id = 0; clip_id < video_num_; clip_id++)
	{
		int start = seidx_data[clip_id*2];
		int end = seidx_data[clip_id*2+1];
		for (int frame_id = start; frame_id <= end; frame_id++)
		{
			caffe_cpu_scale(N_, top_diff[frame_id]/Dtype(2.0), center_data + clip_id*N_, bottom_diff + frame_id*N_);
			if (top.size()>1){
				caffe_axpy(N_, -loss_weight_/M_/Dtype(2.0), center_data + clip_id*N_, bottom_diff + frame_id*N_);
			}
		}		
	}
  }
  if (propagate_down[2]) {
	Dtype* bottom_diff = bottom[2]->mutable_cpu_diff();
	caffe_set(bottom[2]->count(), Dtype(0.), bottom_diff);
    // Gradient with respect to bottom data
	for(int clip_id = 0; clip_id < video_num_; clip_id++)
	{
		int start = seidx_data[clip_id*2];
		int end = seidx_data[clip_id*2+1];
		for (int frame_id = start; frame_id <= end; frame_id++)
		{
			caffe_axpy(N_, top_diff[frame_id]/Dtype(2.0), bottom_data + frame_id*N_, bottom_diff + clip_id*N_);
			if (top.size()>1){
				caffe_axpy(N_, -loss_weight_/M_/Dtype(2.0), bottom_data + frame_id*N_, bottom_diff + clip_id*N_);
			}
		}		
	}
	
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterSimilarityLayer);
#endif

INSTANTIATE_CLASS(CenterSimilarityLayer);
REGISTER_LAYER_CLASS(CenterSimilarity);

}  // namespace caffe
