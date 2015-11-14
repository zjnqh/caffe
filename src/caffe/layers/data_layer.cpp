// #ifdef USE_OPENCV
// #include <opencv2/core/core.hpp>
// #endif  // USE_OPENCV
// #include <stdint.h>

// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/data_layers.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/benchmark.hpp"
// #include "caffe/util/io.hpp"

// //jq add
// #include "boost/scoped_ptr.hpp"
// using boost::scoped_ptr;
// namespace caffe {

// template <typename Dtype>
// DataLayer<Dtype>::DataLayer(const LayerParameter& param)
//   : BasePrefetchingDataLayer<Dtype>(param),
//     reader_(param) {
// }

// template <typename Dtype>
// DataLayer<Dtype>::~DataLayer() {
//   this->StopInternalThread();
// }


// template <typename Dtype>
// void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   // LOG(INFO)<<"jq data_layers setup";
//   const int batch_size = this->layer_param_.data_param().batch_size();
//   // Read a data point, and use it to initialize the top blob.
//   // Datum& datum = *(reader_.full().peek());
//   db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
//   db_->Open(this->layer_param_.data_param().source(), db::WRITE);
//   // LOG(INFO)<<"jq Transaction 1";
//   // scoped_ptr<db::Transaction> txn(db_->NewTransaction());
//   // LOG(INFO)<<"jq Transaction 2";
//   cursor_.reset(db_->NewCursor());
//   // LOG(INFO)<<"jq Transaction 3";
//   // scoped_ptr<db::Transaction> txn2(db_->NewTransaction());
//   // LOG(INFO)<<"jq Transaction 4";
//   Datum datum;
//   datum.ParseFromString(cursor_->value());
//   this->use_data_.Reshape(1,1,1,batch_size + 5);
//   // Use data_transformer to infer the expected blob shape from datum.
//   vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
//   this->transformed_data_.Reshape(top_shape);
//   // Reshape top[0] and prefetch_data according to the batch_size.
//   top_shape[0] = batch_size;
//   top[0]->Reshape(top_shape);
//   for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//     this->prefetch_[i].data_.Reshape(top_shape);
//   }
//   LOG(INFO) << "output data size: " << top[0]->num() << ","
//       << top[0]->channels() << "," << top[0]->height() << ","
//       << top[0]->width();
//   // label
//   if (this->output_labels_) {
//     vector<int> label_shape(1, batch_size);
//     top[1]->Reshape(label_shape);
//     for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//       this->prefetch_[i].label_.Reshape(label_shape);
//     }
//   }
// }

// // This function is called on prefetch thread
// template<typename Dtype>
// void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
//   CPUTimer batch_timer;
//   batch_timer.Start();
//   double read_time = 0;
//   double trans_time = 0;
//   CPUTimer timer;
//   CHECK(batch->data_.count());
//   CHECK(this->transformed_data_.count());

//   // Reshape according to the first datum of each batch
//   // on single input batches allows for inputs of varying dimension.
//   const int batch_size = this->layer_param_.data_param().batch_size();
//   // Datum& datum = *(reader_.full().peek());
//   // // Use data_transformer to infer the expected blob shape from datum.
//   Datum datum;
//   datum.ParseFromString(cursor_->value());
//   // Use data_transformer to infer the expected blob shape from datum.
//   vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
//   // this->transformed_data_.Reshape(top_shape);
//   // Reshape batch according to the batch_size.
//   top_shape[0] = batch_size;
//   batch->data_.Reshape(top_shape);

//   Dtype* top_data = batch->data_.mutable_cpu_data();
//   Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
//   // LOG(INFO)<<"jq here";
//   if (this->output_labels_) {
//     top_label = batch->label_.mutable_cpu_data();
//   }
//   // std::cout<<"jq here";
//   Dtype* use_data=this->use_data_.mutable_cpu_data();
//   // LOG(INFO)<<"jq "<<use_data[0]<<" "<<(use_data[0]==1.0);
//   int disp_size=std::min(10,batch_size);
//   if (use_data[3]!=0.0){
//     cursor_->SeekToFirst();
//     use_data[3]=0.0;
//     use_data[4]=0.0;
//   }
//   if (use_data[0]==0.0){
//     Dtype start= use_data[4];
//     // LOG(INFO)<<"jq use_data=0";
//     // LOG(INFO)<<"jq enter data_layers "<<batch_size<<" "<< batch->data_.shape_string();
//     for (int item_id = 0; item_id < batch_size; item_id++) {
//       // timer.Start();
//       // get a datum
//       // Datum& datum = *(reader_.full().pop("Waiting for data"));
//       Datum datum;
//       datum.ParseFromString(cursor_->value());
   
//       // Apply data transformations (mirror, scale, crop...)
//       // LOG(INFO)<<"jq enter data_layers"<< item_id;
//       int offset = batch->data_.offset(item_id);
//       // LOG(INFO)<<"jq enter data_layers";
//       this->transformed_data_.set_cpu_data(top_data + offset);
//       this->data_transformer_->Transform(datum, &(this->transformed_data_));
//       // Copy label.
//       if (this->output_labels_) {
//         top_label[item_id] = datum.label();
//       }
//       use_data[item_id +5] = start;
//       // trans_time += timer.MicroSeconds();
//       cursor_->Next();
//       start +=1.0;
//       if (!cursor_->valid()) {
//         DLOG(INFO) << "Restarting data prefetching from start.";
//         cursor_->SeekToFirst();
//       }
//       // reader_.free().push(const_cast<Datum*>(&datum));
//     }
//     if (use_data[1]!=0.0){
//       use_data[2]=static_cast<Dtype>(batch_size);
//       use_data[4]+=static_cast<Dtype>(batch_size);
//       LOG(INFO)<<"jq cpp label";
//       for (int item_id = 0; item_id < disp_size; item_id++) {
//         std::cout<<" "<<top_label[item_id];
//       }
//       std::cout<<std::endl;
//       LOG(INFO)<<"jq use_data";
//       for (int item_id = 0; item_id < disp_size; item_id++) {
//         std::cout<<" "<<use_data[item_id +5];
//       }
//       std::cout<<std::endl;
//     }
//   }
//   else if (use_data[0]==1.0){
//     LOG(INFO)<<"jq use_data=1";
//     //extract unlabeled data 
//     int step_over = batch_size;
//     // LOG(INFO)<<"batch_size:"<<batch_size;
//     // int UNLABEL=static_cast<int>(use_data[1]);
//     Dtype UNLABEL=use_data[1];
//     Dtype start = use_data[4];
//     // Dtype datanum = use_data[3];
//     // 0, sami-super-unsuper, 1, label_kinds, 2, step over, 
//     // 3, datanum, 4, start index
//     // LOG(INFO)<<"batch_size:"<<batch_size<<" UNLABEL"<<UNLABEL;
//     // std::cout<<"Start: ";
//     for (int item_id = 0; item_id < batch_size; item_id++) {
//       Datum datum;
//       datum.ParseFromString(cursor_->value());
//       // std::cout<<" "<<datum.label();
//       while(datum.label()<UNLABEL){
//         cursor_->Next();
//         if (!cursor_->valid()) {
//           DLOG(INFO) << "Restarting data prefetching from start.";
//           cursor_->SeekToFirst();
//         }
//         datum.ParseFromString(cursor_->value());
//          // LOG(INFO)<<start<<" stepping over"<<datum.label();
//         start+=1.0;step_over ++;
//       }
//       // std::cout<<datum.label()<<" ";
//       use_data[item_id +5] = start;
//       // read_time += timer.MicroSeconds();
//       int offset = batch->data_.offset(item_id);
//       this->transformed_data_.set_cpu_data(top_data + offset);
//       this->data_transformer_->Transform(datum, &(this->transformed_data_));
//       // Copy label.
//       if (this->output_labels_) {
//         top_label[item_id] = datum.label();
//       }
//       // if(item_id<10)std::cout<<"jq "<<datum.label();
//       // trans_time += timer.MicroSeconds();
//       cursor_->Next();
//       start+=1.0;
//       if (!cursor_->valid()) {
//         DLOG(INFO) << "Restarting data prefetching from start.";
//         cursor_->SeekToFirst();
//       }
//     }
//     LOG(INFO)<<"jq cpp label";
//     for (int item_id = 0; item_id < disp_size; item_id++) {
//       std::cout<<" "<<top_label[item_id];
//     }
//     std::cout<<std::endl;
//     LOG(INFO)<<"jq use_data";
//     for (int item_id = 0; item_id < disp_size; item_id++) {
//       std::cout<<" "<<use_data[item_id +5];
//     }
//     std::cout<<std::endl;
//     use_data[2]=static_cast<Dtype>(step_over);
//     use_data[4]=static_cast<Dtype>(start);
//     // LOG(INFO)<<"cpp step over:"<<step_over;
//   }
//   else if (use_data[0]==2.0){
//     // forward-backward using semi supervised with false label
//     // 0, sami-super-unsuper, 1, label_kinds, 2, step over, 
//     // 3, datanum, 4, start index
//     int step_over = batch_size;
//     // Dtype UNLABEL=use_data[1];
//     Dtype start = use_data[4];
//     // LOG(INFO)<<"jq use_data=2";
//     // Dtype datanum = use_data[3];
//     // for (int i =use_data[4];i<; i++){
//     //   Datum& datum = *(reader_.full().pop("Waiting for data"));
//     //   reader_.free().push(const_cast<Datum*>(&datum));
//     //   // step_over ++;
//     // }
//     // LOG(INFO)<<"batch_size:"<<batch_size;
//     std::cout<<std::endl;
//     for (int item_id = 0; item_id < batch_size; item_id++) {
//       // timer.Start();
//       // get a datum
//       Datum datum ;
//       datum.ParseFromString(cursor_->value());
//       std::cout<<"   "<<start<<":"<<datum.label()<<"  ";
//       // int step= use_data[item_id + 3];
//       // LOG(INFO)<<"start "<<start<< " end "<<use_data[item_id +5];
//       while(start<use_data[item_id +5]){
//         // LOG(INFO)<<start<<" stepping over"<<datum.label();
//         cursor_->Next();
//         start+=1.0;
//         if (!cursor_->valid()) {
//           DLOG(INFO) << "Restarting data prefetching from start.";
//           cursor_->SeekToFirst();
//         }
//         datum.ParseFromString(cursor_->value());
//         std::cout<<"   "<<start<<":"<<datum.label()<<"  ";
//         // start+=1.0;step_over ++;
//       }
      
//       // read_time += timer.MicroSeconds();
//       // start=use_data[item_id +5];
//       timer.Start();
//       // Apply data transformations (mirror, scale, crop...)
//       int offset = batch->data_.offset(item_id);
//       this->transformed_data_.set_cpu_data(top_data + offset);
//       this->data_transformer_->Transform(datum, &(this->transformed_data_));
//       // Copy label.
//       if (this->output_labels_) {
//         top_label[item_id] = datum.label();
//         // top_label[item_id]= static_cast<int>(use_data[item_id + batch_size +3]);
//       }
//       trans_time += timer.MicroSeconds();
//       cursor_->Next();
//       start+=1.0;
//       if (!cursor_->valid()) {
//         DLOG(INFO) << "Restarting data prefetching from start.";
//         cursor_->SeekToFirst();
//       }
//     }
//     //jq test key value access
//     LOG(INFO)<<"jq Transaction start";
//     scoped_ptr<db::Transaction> txn(db_->NewTransaction());
//     LOG(INFO)<<"jq Transaction done";
//     int kCIFARImageNBytes=32;
//     for (int item_id = 0; item_id < batch_size; item_id++) {
//       char str_buffer[kCIFARImageNBytes];
//       int id= static_cast<int>(use_data[item_id+ 5]);
//       int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", id);
//       string value;
//       string str=string(str_buffer, length);
//       txn->Get(str, value);
//       Datum datum;
//       datum.ParseFromString(value);
//       if (this->output_labels_) {
//         top_label[item_id] = datum.label();
//         // top_label[item_id]= static_cast<int>(use_data[item_id + batch_size +3]);
//       }
//     }
//     txn->Commit();
//     std::cout<<std::endl;
//     use_data[4]=static_cast<Dtype>(start );
//     LOG(INFO)<<"jq cpp label";
//     for (int item_id = 0; item_id < disp_size; item_id++) {
//       std::cout<<" "<<top_label[item_id];
//     }
//     std::cout<<std::endl;
//     LOG(INFO)<<"jq use_data";
//     for (int item_id = 0; item_id < disp_size; item_id++) {
//       std::cout<<" "<<use_data[item_id +5];
//     }
//     std::cout<<std::endl;
//   }
//   timer.Stop();
//   batch_timer.Stop();
//   DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//   DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//   DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
// }

// INSTANTIATE_CLASS(DataLayer);
// REGISTER_LAYER_CLASS(Data);

// }  // namespace caffe




#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

//jq add
#include "boost/scoped_ptr.hpp"
using boost::scoped_ptr;
namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}


template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LOG(INFO)<<"jq data_layers setup";
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  // Datum& datum = *(reader_.full().peek());
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::WRITE);
  // LOG(INFO)<<"jq Transaction 1";
  // scoped_ptr<db::Transaction> txn(db_->NewTransaction());
  // LOG(INFO)<<"jq Transaction 2";
  cursor_.reset(db_->NewCursor());
  // LOG(INFO)<<"jq Transaction 3";
  // scoped_ptr<db::Transaction> txn2(db_->NewTransaction());
  // LOG(INFO)<<"jq Transaction 4";
  Datum datum;
  datum.ParseFromString(cursor_->value());
  this->use_data_.Reshape(1,1,1,batch_size+ batch_size + 1);
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  this->jqbatch_shape = this->data_transformer_->InferBlobShape(datum);
  this->jqbatch_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  // LOG(INFO)<<"load_batch";
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Datum& datum = *(reader_.full().peek());
  // // Use data_transformer to infer the expected blob shape from datum.
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  // Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  // LOG(INFO)<<" output_labels_:"<<this->output_labels_;
  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
    // top_label = this->prefetch_label_.mutable_cpu_data();
  }
  Dtype* use_data=this->use_data_.mutable_cpu_data();
  // LOG(INFO)<<" use_data[0]:"<<use_data[0];
  if (use_data[0]==0.0){
    // LOG(INFO)<<"visit in order";
    for (int item_id = 0; item_id < batch_size; item_id++) {
      Datum datum;
      datum.ParseFromString(cursor_->value());
   
      // Apply data transformations (mirror, scale, crop...)
      // LOG(INFO)<<"jq enter data_layers"<< item_id;
      int offset = batch->data_.offset(item_id);
      // LOG(INFO)<<"jq enter data_layers";
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      // Copy label.
      if (this->output_labels_) {
        top_label[item_id] = datum.label();
        // std::cout<<" cursor_:"<<datum.label();
      }
      // use_data[item_id +5] = start;
      // trans_time += timer.MicroSeconds();
      cursor_->Next();
      // start +=1.0;
      // std::cout<<" output_labels_:"<<this->output_labels_;
      if (!cursor_->valid()) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();
      }
      // reader_.free().push(const_cast<Datum*>(&datum));
    }
  }else if (use_data[0]!=0.0){
    // forward-backward using semi supervised with false label
    // 0, sami-super-unsuper, 1, label_kinds, 2, step over, 
    // 3, datanum, 4, start index
    // LOG(INFO)<<"visit in Key/value";
    // LOG(INFO)<<"this->PREFETCH_COUNT:"<<this->PREFETCH_COUNT;
    int step_over = batch_size+1;
    // std::cout<<std::endl;
    scoped_ptr<db::Transaction> txn(db_->NewTransaction());
    // std::cout<<"key:";
    int kCIFARImageNBytes=3072;
    for (int item_id = 0; item_id < batch_size; item_id++) {
      char str_buffer[kCIFARImageNBytes];
      int id= static_cast<int>(use_data[item_id+ 1]);
      // std::cout<<" "<<id<<":";
      int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", id);
      string value;
      string str=string(str_buffer, length);
      txn->Get(str, value);
      Datum datum;
      datum.ParseFromString(value);
      int offset = batch->data_.offset(item_id);
      // LOG(INFO)<<"jq enter data_layers";
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      // std::cout<<" output_labels_:"<<this->output_labels_;
      if (this->output_labels_) {
        // top_label[item_id] = datum.label();
        top_label[item_id] = use_data[item_id+ step_over];
        // std::cout<<" KV:"<<datum.label();
        // top_label[item_id]= static_cast<int>(use_data[item_id + batch_size +3]);
      }
      if( use_data[item_id+ step_over]!=(datum.label()%1000))
        LOG(INFO)<<"image id:"<<id<<" not correctly fetch: "<<datum.label()
      <<" vs "<<use_data[item_id+ step_over];
      // std::cout<<top_label[item_id];
      // std::cout<<" key:"<<id;
    }
    // std::cout<<std::endl;
    // for (int item_id = 0; item_id < 50000; item_id++) {
    //   char str_buffer[kCIFARImageNBytes];
    //   // int id= static_cast<int>(use_data[item_id+ 1]);
    //   int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", item_id);
    //   string value;
    //   string str=string(str_buffer, length);
    //   txn->Get(str, value);
    //   // Datum datum;
    //   // datum.ParseFromString(value);
    //   // int offset = batch->data_.offset(item_id);
    //   // // LOG(INFO)<<"jq enter data_layers";
    //   // this->transformed_data_.set_cpu_data(top_data + offset);
    //   // this->data_transformer_->Transform(datum, &(this->transformed_data_));
    //   // if (this->output_labels_) {
    //   //   top_label[item_id] = datum.label();
    //   //   // top_label[item_id]= static_cast<int>(use_data[item_id + batch_size +3]);
    //   // }
    //   // std::cout<<" "<<item_id;
    // }
    // std::cout<<std::endl;
    txn->Commit();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
// void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);





// #ifdef USE_OPENCV
// #include <opencv2/core/core.hpp>
// #endif  // USE_OPENCV
// #include <stdint.h>

// #include <string>
// #include <vector>

// #include "caffe/common.hpp"
// #include "caffe/data_layers.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/benchmark.hpp"
// #include "caffe/util/io.hpp"

// namespace caffe {

// template <typename Dtype>
// DataLayer<Dtype>::DataLayer(const LayerParameter& param)
//   : BasePrefetchingDataLayer<Dtype>(param),
//     reader_(param) {
// }

// template <typename Dtype>
// DataLayer<Dtype>::~DataLayer() {
//   this->StopInternalThread();
// }

// template <typename Dtype>
// void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   const int batch_size = this->layer_param_.data_param().batch_size();
//   // Read a data point, and use it to initialize the top blob.
//   Datum& datum = *(reader_.full().peek());

//   // Use data_transformer to infer the expected blob shape from datum.
//   vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
//   this->transformed_data_.Reshape(top_shape);
//   // Reshape top[0] and prefetch_data according to the batch_size.
//   top_shape[0] = batch_size;
//   top[0]->Reshape(top_shape);
//   for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//     this->prefetch_[i].data_.Reshape(top_shape);
//   }
//   LOG(INFO) << "output data size: " << top[0]->num() << ","
//       << top[0]->channels() << "," << top[0]->height() << ","
//       << top[0]->width();
//   // label
//   if (this->output_labels_) {
//     vector<int> label_shape(1, batch_size);
//     top[1]->Reshape(label_shape);
//     for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//       this->prefetch_[i].label_.Reshape(label_shape);
//     }
//   }
// }

// // This function is called on prefetch thread
// template<typename Dtype>
// void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
//   CPUTimer batch_timer;
//   batch_timer.Start();
//   double read_time = 0;
//   double trans_time = 0;
//   CPUTimer timer;
//   CHECK(batch->data_.count());
//   CHECK(this->transformed_data_.count());
  
//   // Reshape according to the first datum of each batch
//   // on single input batches allows for inputs of varying dimension.
//   const int batch_size = this->layer_param_.data_param().batch_size();
//   Datum& datum = *(reader_.full().peek());
//   // Use data_transformer to infer the expected blob shape from datum.
//   vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
//   this->transformed_data_.Reshape(top_shape);
//   // Reshape batch according to the batch_size.
//   top_shape[0] = batch_size;
//   batch->data_.Reshape(top_shape);

//   Dtype* top_data = batch->data_.mutable_cpu_data();
//   Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
//   // LOG(INFO)<<"jq load_batch batch_size "<<batch_size;
//   if (this->output_labels_) {
//     top_label = batch->label_.mutable_cpu_data();
//   }
//   int jqid=0;
//   for (int item_id = 0; item_id < batch_size; ++item_id) {
//     timer.Start();
//     // get a datum
//     Datum& datum = *(reader_.full().pop("Waiting for data"));
//     read_time += timer.MicroSeconds();
//     // if(jqid++ % 2==0) {
//     timer.Start();
//     // Apply data transformations (mirror, scale, crop...)
//     int offset = batch->data_.offset(item_id);
//     this->transformed_data_.set_cpu_data(top_data + offset);
//     this->data_transformer_->Transform(datum, &(this->transformed_data_));
//     // Copy label.
//     if (this->output_labels_) {
//       top_label[item_id] = datum.label();
//     }
//     // std::cout<<"jq "<<item_id;
//     trans_time += timer.MicroSeconds();
//     reader_.free().push(const_cast<Datum*>(&datum));
//   }
//   timer.Stop();
//   batch_timer.Stop();
//   DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//   DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//   DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
// }

// INSTANTIATE_CLASS(DataLayer);
// REGISTER_LAYER_CLASS(Data);

// }  // namespace caffe
