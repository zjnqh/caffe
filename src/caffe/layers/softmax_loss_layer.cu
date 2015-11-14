#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if ((has_ignore_label_ && label_value == ignore_label_) ||(label_value>= dim) ){
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU_semi(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,const Dtype* used,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (used[n * spatial_dim + s+1]==0) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  // LOG(INFO)<<"jq label shape"<<bottom[1]->shape_string();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  // LOG(INFO)<<"jq Forward start";
  if (
    false
    // bottom[1]->count()==(this->use_data_.count())
    ){
    const Dtype* used=this->use_data_.gpu_data();
    SoftmaxLossForwardGPU_semi<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,used, counts);
  }
  else{
    SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  }
  // LOG(INFO)<<"jq Forward end";
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    if (count==0)count=1;
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
  prob_data = prob_.cpu_data();
  // LOG(INFO)<<" prob_data: "<<sizeof((prob_data));
  // for (int n=0;n<3;n++)std::cout<<prob_data[0]<<" ";

}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU_semi(const int nthreads, const Dtype* prob_data,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_,const Dtype* used, Dtype* counts) {
  const int channels = dim / spatial_dim;
  //spatial_dim= 1, dim number of classes, num:batchsize, nthreads: batchsize* spatial_dim
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    
    // Dtype optimum = -100000.0;
    // int pivot=0;
    // for (int c = 0; c < channels; ++c) {
    //   if (prob_data[n * dim + c * spatial_dim + s]>optimum){
    //     optimum=prob_data[n * dim + c * spatial_dim + s];
    //     pivot=c;
    //   }
    // }
    // if (used[n * spatial_dim + s+1]==1.0) {
    // // if (used[index]==0) {
    //   for (int c = 0; c < channels; ++c) {
    //     bottom_diff[n * dim + c * spatial_dim + s] = 0;
    //   }
    //   counts[index] = 0;
    // } else if (used[n * spatial_dim + s+1]==0.0){
    //   const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    //   bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
    //   counts[index] = 1;
    // }else{
    //   int label_value = static_cast<int>(used[n * spatial_dim + s+1])-2;
    //   bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
    //   counts[index] = 1;
    //   // counts[index] = 0;
      
    // }
    if (used[n * spatial_dim + s+1]>=1000.0) {
    // if (used[index]==0) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      const int label_value = static_cast<int>(used[n * spatial_dim + s +1]);
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    // LOG(INFO)<<" prob_diff: "<<sizeof((prob_.gpu_diff()));
    // LOG(INFO)<<" bottom_diff: "<<sizeof((bottom[0]->gpu_diff()));
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // LOG(INFO)<<"dim "<<dim<<" outer_num_ "<< outer_num_<< " inner_num_ "<<inner_num_;
    //jq note: dim: number of classes, outer_num_: batchsize, inner_num_:1
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    // int countjq=0;
    // const Dtype* used_=this->use_data_.cpu_data();
    // for(int i=0;i<nthreads;i++) {if(used_[i]==0) countjq++;}
    // LOG(INFO)<<"jq used index"<<countjq;
    const Dtype* semi_info=this->use_data_.cpu_data();
    // const Dtype* label_info=bottom[1]->cpu_data();
    if (
      // false
      // bottom[1]->count()==(this->use_data_.count())
      semi_info[0]!=0.0
      ){
      // LOG(INFO)<<"using semi loss";
      // LOG(INFO)<<"jq using SoftmaxLossBackwardGPU_semi";
      // int jqcount=0;
      // for (int i =0;i< outer_num_;i++){
      //   int cp=static_cast<int>(label_info[i]-1000);
      //   if (cp ==static_cast<int>(semi_info[i+1]-2))jqcount++;
      // }
      // LOG(INFO)<<" jqcount:"<<jqcount;
      const Dtype* used=this->use_data_.gpu_data();
      SoftmaxLossBackwardGPU_semi<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,used, counts);
    }else{
      // LOG(INFO)<<"jq using SoftmaxLossBackwardGPU regular";
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,counts);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // Dtype countjq;
    // caffe_gpu_asum(nthreads, counts, &countjq);
    // LOG(INFO)<<"count "<<countjq;
    // Dtype accurate;
    // caffe_gpu_asum(nthreads, counts, &accurate);
    // LOG(INFO)<<"count "<<accurate;
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      if (count==0)count=1;
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe

