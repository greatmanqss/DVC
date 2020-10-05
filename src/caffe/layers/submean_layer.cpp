#include <cfloat>
#include <vector>
#include <fstream>
#include "caffe/layers/submean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype> 
void SubmeanLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    CHECK(this->layer_param().submean_param().clips_num() > 0) <<
	"Submean Layer takes at least one video clips.";
    video_num_ = this->layer_param().submean_param().clips_num();
}
 
template <typename Dtype> 
void SubmeanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    
	if(top.size()>2)//label info 
        top[2]->Reshape(video_num_,1,1,1);
	if(top.size()>3)
	{
		//videos 
		top[0]->Reshape(bottom[0]->num()-video_num_,bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
		//still images
		top[3]->Reshape(video_num_,bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
		top[4]->Reshape(video_num_,1,1,1);
	}
    else{
		top[0]->ReshapeLike(*bottom[0]);
	}	

    start_end_idx_.Reshape(video_num_,2,1,1);
	top[1]->Reshape(video_num_,2,1,1);
    //differ_.Reshape(1,bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    mean_.Reshape(video_num_, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SubmeanLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			//fstream output;
			//output.open("/home/data/qiaoshishi/index.txt",ios::out);
			//output<<"batch:"<<std::endl;
	//int batch_size = bottom[1]->num();//qss
    Dtype  first_label = bottom[1]->cpu_data()[1];//qss
	//output<<first_label<<'\t';
    Dtype* seidx_data = start_end_idx_.mutable_cpu_data();
    //Dtype* mean_data = mean_.mutable_cpu_data();
    Dtype* label_data = NULL;
    if(top.size()>2){
        label_data = top[2]->mutable_cpu_data();
        label_data[0] = bottom[1]->cpu_data()[0];
		//output<<label_data[0]<<'\t';
    }

    // dim of frame feature
    //const int count = bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
    //caffe_set(mean_.count(), Dtype(0), mean_data);

    seidx_data[0] = 0;
	//output<<seidx_data[0]<<'\t';
    int clip_count = 1;
    //int offset_mean = mean_.offset(clip_count-1);
    //int offset_bottom = bottom[0]->offset(0);
    // sum each clip's frame features
    //caffe_axpy(count, Dtype(1.0), bottom[0]->cpu_data()+offset_bottom, mean_data+offset_mean);
    // calaculate m = 1/n * sum (fi) for each clip
    for(int i = 1; i < bottom[1]->num(); i++)
    {
        seidx_data[clip_count*2-1] = i;
        if(bottom[1]->cpu_data()[i*2+1] != first_label)// the end of the previous clip and begin of a new clip qss
        {
            seidx_data[clip_count*2-1] = (top.size()>3) ? (i-clip_count):(i-1);// fix the end index of previous clip in top[0]
			//output<<seidx_data[clip_count*2-1]<<std::endl;
			if(top.size()>3){        
              top[4]->mutable_cpu_data()[clip_count-1] = bottom[1]->cpu_data()[i*2];//qss
            }			
			
            //Dtype coeff  = 1.0/(seidx_data[clip_count*2-1] -seidx_data[clip_count*2-2] + 1);
            // average frame features of previous clip
            //caffe_scal(count, coeff, mean_data+offset_mean);
            if (clip_count < video_num_)// config the new clip
            {
                seidx_data[clip_count*2] = seidx_data[clip_count*2-1]+1;
				if(top.size()>3){//skip the still image
					i = i + 1;
				}
                first_label = bottom[1]->cpu_data()[i*2+1];//qss
				//output<<first_label<<'\t';
                if(top.size()>2) {
                    label_data[clip_count] = bottom[1]->cpu_data()[i*2];//qss
					//output<<label_data[clip_count]<<'\t';
                }
				//output<<seidx_data[clip_count*2]<<'\t';
                clip_count ++;
                //offset_mean = mean_.offset(clip_count-1);
            }
            else// all clips have been found 
            {
                break;
            }
        }
        //offset_bottom = bottom[0]->offset(i);
        //caffe_axpy(count, Dtype(1.0), bottom[0]->cpu_data()+offset_bottom, mean_data+offset_mean);
    }
	//output.close();
	// the for loop replaces the /*...*/ above for DVC journal weighted temporal average pooling
	/*for(int i=0; i<video_num_; i++){
		label_data[i] = bottom[1]->cpu_data()[i*20];
		seidx_data[i*2] = i*10;
		seidx_data[i*2+1] = (i+1)*10-1; 
	}*/
	caffe_copy(top[1]->count(),seidx_data,top[1]->mutable_cpu_data());
	if(top.size()<=3)
	   caffe_copy(top[0]->count(),bottom[0]->cpu_data(),top[0]->mutable_cpu_data());
    else
	{
		for(int clip_id = 0; clip_id < video_num_; clip_id++)
      {
        int start = start_end_idx_.cpu_data()[clip_id*2];
        int end = start_end_idx_.cpu_data()[clip_id*2+1];
        int clip_size = end - start + 1;
        int offset_v = bottom[0]->offset(start+clip_id);
		int offset_s = bottom[0]->offset(end+clip_id+1);
		int offset_v_t = top[0]->offset(start);
		int offset_s_t = top[3]->offset(clip_id);
		caffe_copy(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()*clip_size, 
		                    bottom[0]->cpu_data()+offset_v, top[0]->mutable_cpu_data()+offset_v_t);
		caffe_copy(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(),
		                    bottom[0]->cpu_data()+offset_s, top[3]->mutable_cpu_data()+offset_s_t);
      }
	}
	
}

template <typename Dtype>
void SubmeanLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    for (int i = 0; i < bottom.size(); ++i) {
        if (propagate_down[i]) {
			//const int count = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
            const Dtype* top_diff_v = top[0]->cpu_diff();
			    
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			caffe_set(bottom[i]->count(),Dtype(0.),bottom_diff);
						
			if(top.size()<=3)//only video modal
	          caffe_copy(bottom[i]->count(), top_diff_v, bottom_diff);
            else//cross-modal
	       {
			  const Dtype* top_diff_s = top[3]->cpu_diff();
		      for(int clip_id = 0; clip_id < video_num_; clip_id++)
             {
                int start = start_end_idx_.cpu_data()[clip_id*2];
                int end = start_end_idx_.cpu_data()[clip_id*2+1];
                int clip_size = end - start + 1;
                int offset_v = bottom[i]->offset(start+clip_id);
		        int offset_s = bottom[i]->offset(end+clip_id+1);
		        int offset_v_t = top[0]->offset(start);
		        int offset_s_t = top[3]->offset(clip_id);
		        caffe_copy(bottom[i]->channels()*bottom[i]->height()*bottom[i]->width()*clip_size, 
		                     top_diff_v + offset_v_t, bottom_diff + offset_v);
		        caffe_copy(bottom[i]->channels()*bottom[i]->height()*bottom[i]->width(),
		                     top_diff_s + offset_s_t, bottom_diff + offset_s);
             }
	       } 

        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SubmeanLayer);
#endif

INSTANTIATE_CLASS(SubmeanLayer);
REGISTER_LAYER_CLASS(Submean);

}  // namespace caffe
