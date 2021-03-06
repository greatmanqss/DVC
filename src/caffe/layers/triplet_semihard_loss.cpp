#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/triplet_semihard_loss.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void TripletSemihardLossLayer<Dtype>::Reshape(
					    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
  }


  //template <typename Dtype>
  //Dtype TripletRankLossLayer<Dtype>::Similarity(
  //						const Dtype* labels, const int i, const int j, const int dim ) {
  //  Dtype s_sim = 0;
  //  for(int k=0; k < dim; k++){
  //    if(labels[i*dim + k] > 0 && labels[i*dim + k] == labels[j*dim + k]){
  //	s_sim++;
  //      return s_sim;
  //    }
  //  }
  //  return s_sim;
  //}


  template <typename Dtype>
  void TripletSemihardLossLayer<Dtype>::Forward_cpu(
						const vector<Blob<Dtype>*>& bottom,
						const  vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->channels();
	
    //int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype margin = this->layer_param_.triplet_rank_loss_param().margin();
	bool sim;
    Dtype loss(0.0);
    Dtype  n_tri = 0;
	
    for (int i = 0; i < num; ++i) {
      for (int j = i+1; j < num; ++j) {
        sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[j])));
		if(!sim)
			continue;
		//Dtype most_difficult = 0.0;
        for (int k = j+1; k < num; ++k) {
			sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[k])));
			if(sim)
				continue;	  
	        Dtype norm1=0, norm2 = 0;
	        for(int l=0; l < dim; ++l){
	            norm1 += pow((bottom_data[i*dim + l] - bottom_data[j*dim + l]),2);
	            norm2 += pow((bottom_data[i*dim + l] - bottom_data[k*dim + l]),2);
	        }
			/* if(margin + norm1 - norm2 > most_difficult){
				most_difficult = margin + norm1 - norm2;
			} */
	        if((margin + norm1 - norm2 > 0) && (norm1 < norm2)){
				n_tri++;
				loss += (margin + norm1 - norm2);
	        }
	    }
		/* if(most_difficult > 0.0){
			n_tri++;
			loss += most_difficult;
		} */
		
      }
	  
    }
	if(n_tri > 0)
	   loss = loss / n_tri;
   
    top[0]->mutable_cpu_data()[0] = loss;
  }

  template <typename Dtype>
  void TripletSemihardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, 0, bottom[0]->count()*sizeof(Dtype));
    const int num = bottom[0]->num();
    const int dim = bottom[0]->channels();
    //int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype margin = this->layer_param_.triplet_rank_loss_param().margin();
	bool sim;
    Dtype  n_tri = 0;

    for (int i = 0; i < num; ++i) {
      for (int j = i+1; j < num; ++j) {
	    sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[j])));
		if(!sim)
			continue;
        //Dtype most_difficult = 0.0;
		//int index = -1;
	    for (int k = j+1; k < num; ++k) {
	        sim = ((static_cast<int>(label[i])) == (static_cast<int>(label[k])));
			if(sim)
				continue;	  
	 
	        Dtype norm1=0, norm2 = 0;
	        for(int l=0; l < dim; ++l){
	            norm1 += pow((bottom_data[i*dim + l] - bottom_data[j*dim + l]),2);
	            norm2 += pow((bottom_data[i*dim + l] - bottom_data[k*dim + l]),2);
	        }
			/* if(margin + norm1 - norm2 > most_difficult){
				most_difficult = margin + norm1 - norm2;
				index = k;
			} */
	        if((margin + norm1 - norm2 > 0) && (norm1 < norm2)){
				n_tri++;
	            for(int l=0; l < dim; ++l){
	               diff[i*dim + l] += 2*(bottom_data[k*dim + l] - bottom_data[j*dim + l]);
	               diff[j*dim + l] += 2*(bottom_data[j*dim + l] - bottom_data[i*dim + l]);
	               diff[k*dim + l] += 2*(bottom_data[i*dim + l] - bottom_data[k*dim + l]); 
	            }
	        }
	    }
		/* if(most_difficult > 0.0 && index != -1){
			n_tri++;
			for(int l=0; l < dim; ++l){
	               diff[i*dim + l] += 2*(bottom_data[index*dim + l] - bottom_data[j*dim + l]);
	               diff[j*dim + l] += 2*(bottom_data[j*dim + l] - bottom_data[i*dim + l]);
	               diff[index*dim + l] += 2*(bottom_data[i*dim + l] - bottom_data[index*dim + l]); 
	            }
		} */
      } 
	  
    }
  
    // Scale down gradient
    if(n_tri > 0){
      caffe_scal(bottom[0]->count(), Dtype(1) / n_tri / margin, diff);
	}
	
  }


#ifdef CPU_ONLY
    STUB_GPU(TripletSemihardLossLayer);
#endif

    INSTANTIATE_CLASS(TripletSemihardLossLayer);
    REGISTER_LAYER_CLASS(TripletSemihardLoss);

}  // namespace caffe
