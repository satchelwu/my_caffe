#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <cmath>



#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"


using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using cv::Mat;

DEFINE_string(model, "./model/5.caffemodel",
    "The save path of trained model to test.");
DEFINE_string(imgs_list, "./filename.txt",
    "The text file containing file path of images to be compressed.");
DEFINE_string(save_dir, "./output_images",
    "The compressed images save position.");
DEFINE_int32(batch_size, 20,
    "The number of images to process in a mini-batch.");
DEFINE_int32(img_width, 696,
    "The width of images to be compressed (must be divided by 8).");
DEFINE_int32(img_height, 496,
    "The height of images to be compressed (must be divided by 8).");


size_t get_img_count(string &file_list, vector<string> &img_files) {
    size_t cnt = 0;
    std::ifstream infile(file_list.c_str());
    string line;
    while (std::getline(infile, line)) {
        img_files.push_back(line);
        cnt++;
    }
    return cnt;
}

unsigned char convert_one_pixel(const float &x) {
    int t = static_cast<int>(x * 127.5 + 127.5);
    if (t < 0) t = 0;
    else if(t > 255) t = 255;
    return static_cast<unsigned char>(t);
}


void calc_stat(Mat &in, Mat &out, vector<float> &psnr, vector<float> &mse) {
    if (in.rows != out.rows || in.cols != out.cols) {
        LOG(ERROR) << "Can't calc stats of two different sized images!" << std::endl;
        return;
    }
    int rows = in.rows, cols = in.cols;
    float cur_psnr = 0, cur_mse = 0, tmp;
    for (int h = 0; h < rows; h++) {
        const uchar* in_ptr = in.ptr<uchar>(h), *out_ptr = out.ptr<uchar>(h);
        int in_idx = 0, out_idx = 0;
        for (int w = 0; w < cols; w++) {
            for (int c = 0; c < 3; c++) {
                tmp = static_cast<float>(in_ptr[in_idx++]) - static_cast<float>(out_ptr[out_idx++]);
                tmp *= tmp;
                cur_mse += tmp;
            }
        }
    }
    cur_mse /= (rows * cols * 3);
    cur_psnr = 10 * log10(255 * 255 / cur_mse);
    LOG(INFO) << "PSNR: " << cur_psnr << "; MSE: " << cur_mse << std::endl; 
    psnr.push_back(cur_psnr);
    mse.push_back(cur_mse);
}

void save_compressed_imgs(int &id, Net<float> *net, size_t last, vector<string> &img_files, vector<float> &psnr, vector<float> &mse) {
    const shared_ptr<Blob<float> > compressed_imgs_blob = net->blob_by_name("pdata");
    const float *data = compressed_imgs_blob->cpu_data();
    int offset = 0;
    int spatial_size = FLAGS_img_width * FLAGS_img_height;
    unsigned char img[FLAGS_img_height][FLAGS_img_width][3];

    for (int n = 0; n < FLAGS_batch_size; n++) {
        offset = compressed_imgs_blob->offset(n);
        for (int h = 0; h < FLAGS_img_height; h++) {
            for (int w = 0; w < FLAGS_img_width; w++) {
                for (int c = 0; c < 3; c++) {
                    img[h][w][c] = convert_one_pixel(*(data + offset + spatial_size * c + h * FLAGS_img_width + w));
                }
            }
        }
        Mat out_img = Mat(FLAGS_img_height, FLAGS_img_width, CV_8UC3, img);
        string output_filename = caffe::format_int(id, 5) + ".png";


        LOG(INFO) << "Processing image file " << img_files[id-1] << "." <<std::endl; 
        
        Mat in_img = caffe::ReadImageToCVMat(img_files[id-1], FLAGS_img_height, FLAGS_img_width, true);
        calc_stat(in_img, out_img, psnr, mse);
        
        LOG(INFO) << "Saving the " << id++ << "th image ...... " << std::endl << std::endl;
        cv::imwrite(FLAGS_save_dir + output_filename, out_img);
        if (id > last) {
            float avg_psnr = 0, avg_mse = 0;
            for (int i = 0; i < last; i++) {
                avg_psnr += psnr[i];
                avg_mse += mse[i];
            }
            avg_psnr /= last;
            avg_mse /= last;

            int ratio_id = net->param_names_index().at("rate");
            Blob<float>* ratio_blob = net->params()[ratio_id].get();
            LOG(INFO) << "AVG RATE: " << *(ratio_blob->cpu_data()) / 12 + 1.0 / 48<< std::endl;
            LOG(INFO) << "AVG PSNR: " << avg_psnr << "; AVG MSE: " << avg_mse << std::endl;
            LOG(INFO) << "Test Done" << std::endl;
            break;
        }
    }
}

void load_batch(vector<string> &img_files, int &id, Net<float> *net, size_t last) {
    // dim1 is batch_size of a batch
    int dim1 = last - id + 1 > FLAGS_batch_size ? FLAGS_batch_size : last - id + 1;
    vector<int> shape(4);
    shape[0] = dim1;
    shape[1] = 3;
    shape[2] = FLAGS_img_height;
    shape[3] = FLAGS_img_width;
    const shared_ptr<Blob<float> > batch = net->blob_by_name("data");
    batch->Reshape(shape);
    float *data = batch->mutable_cpu_data();

    int spatial_size = FLAGS_img_height * FLAGS_img_width;
    float tmp;
   
    int end_point = (last + 1 > id + FLAGS_batch_size) ? (last + 1) : (id + FLAGS_batch_size); 
    for (int i = id; i < end_point; i++) {
        int num_offset = batch->offset(i - id);
        LOG(INFO) << "Reading image " << "[" << i - 1 << "]"  << img_files[i - 1] << std::endl; 
        Mat in_img = caffe::ReadImageToCVMat(img_files[i-1], FLAGS_img_height, FLAGS_img_width, true);
        for (int h = 0; h < FLAGS_img_height; h++) {
            const uchar* in_ptr = in_img.ptr<uchar>(h);
            int in_idx = 0;
            for (int w = 0; w < FLAGS_img_width; w++) {
                float *spatial_data = data + num_offset + h * FLAGS_img_width + w;
                for (int c = 0; c < 3; c++) {
                    tmp = static_cast<float>(in_ptr[in_idx++]);
                    tmp = (tmp - 127.5) / 127.5;
                    float *pixel = spatial_data + c * spatial_size;
                    *pixel = tmp;
                }
            }
        }
    }

    LOG(INFO) << "Load a batch of size " << dim1 << "." << std::endl;
}


void get_gpus(vector<int>* gpus) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        gpus->push_back(i);
    }
}


void set_device() {
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() > 0) {
        caffe::Caffe::SetDevice(gpus[0]);   // default to use the 1th gpu device
    }
}

int main(int argc, char** argv) {
    caffe::GlobalInit(&argc, &argv);
    FLAGS_logtostderr = 1;

    // set_device();
    Caffe::set_mode(Caffe::GPU);    // use GPU

    caffe::NetParameter net_param;

    ReadNetParamsFromTextFileOrDie("/home/snk/bin/caffe-master/examples/image_compression/model_rc/deploy.prototxt", &net_param);
    caffe::ImageDataParameter* image_data_param = net_param.mutable_layer(0)->mutable_image_data_param();

    size_t imgs_cnt = 0;
    vector<string> img_files;

    vector<float> psnr, mse;
    
    if (FLAGS_imgs_list.size() > 0) {
        image_data_param->set_source(FLAGS_imgs_list);
        imgs_cnt = get_img_count(FLAGS_imgs_list, img_files);
        LOG(INFO) << "Load image data from " << FLAGS_imgs_list << std::endl;        
    } else {
        LOG(ERROR) << "You must specify imgs_list param!" << std::endl;
    }

    if (FLAGS_batch_size > 0) {
        image_data_param->set_batch_size(FLAGS_batch_size);
        LOG(INFO) << "Set batch size to " << FLAGS_batch_size << std::endl;
    }

    if (FLAGS_img_width > 0 ) {
        if (FLAGS_img_width % 8 != 0) {
            FLAGS_img_width -= FLAGS_img_width % 8;
            LOG(INFO) << "Change your setting img_width to " << FLAGS_img_width << ", because it is not divided by 8." << std::endl;
        }
        image_data_param->set_new_width(FLAGS_img_width);
        LOG(INFO) << "Set image width to " << FLAGS_img_width << std::endl;
    }

    if (FLAGS_img_height > 0 ) {
        if (FLAGS_img_height % 8 != 0) {
            FLAGS_img_height -= FLAGS_img_height % 8;
            LOG(INFO) << "Change your setting img_height to " << FLAGS_img_height << ", because it is not divided by 8." << std::endl;
        }
        image_data_param->set_new_height(FLAGS_img_height);
        LOG(INFO) << "Set image height to " << FLAGS_img_height << std::endl;
    }

    if (FLAGS_save_dir.size() > 0) {
        int len = FLAGS_save_dir.size();
        if(FLAGS_save_dir[len-1] != '/') {  // ensure the last char is /
            FLAGS_save_dir += '/';
        }
    }


    shared_ptr<Net<float> > trained_net(new Net<float>(net_param));
    if (FLAGS_model.size() > 0) {
        trained_net->CopyTrainedLayersFrom(FLAGS_model);
        LOG(INFO) << "Load pretrained model weight successful!" << std::endl;
    } else {
        LOG(ERROR) << "You must specify model param!" << std::endl;        
    }

    int iter_num = imgs_cnt / FLAGS_batch_size;
    bool remainder_ = imgs_cnt % FLAGS_batch_size;
    
    int out_id = 1;
    double time_per_batch, total_time = 0; 
    Timer timer;
    for (size_t i = 0; i < iter_num + remainder_; i++) {
        LOG(INFO) << "Processing the " << i << "th batch data ...... " << std::endl;
        LOG(INFO) << "Loading data ...... " << std::endl;
        load_batch(img_files, out_id, trained_net.get(), imgs_cnt);
        timer.Start();
        LOG(INFO) << "Forward Pass Start!" << std::endl;
        trained_net->ForwardFrom(1);    // skip data forward()
        LOG(INFO) << "Forward Pass Over!" << std::endl;
        time_per_batch = timer.MilliSeconds();
        total_time += time_per_batch;
        LOG(INFO) << "Batch processing time is " << time_per_batch << "ms." << std::endl;
        save_compressed_imgs(out_id, trained_net.get(), imgs_cnt, img_files, psnr, mse);
    }

    LOG(INFO) << "Average image processing time is " << total_time / imgs_cnt << "ms." << std::endl;

    return 0;
}