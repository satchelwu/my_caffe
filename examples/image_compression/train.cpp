#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

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



/******
*  ex:
*       train.bin --save_path=/home/snk/v1 --iter=100000 --rate=0.5
*
*******/
DEFINE_string(save_path, "",
    "The trained model binary file save path.");
DEFINE_int32(iter, -1,
    "The number of iterations to train.");
DEFINE_double(rate, -1,
    "Compression rate threshold to penalize");

bool changeSavePath(caffe::SolverParameter &solver_param, string save_path) {
    solver_param.set_snapshot_prefix(save_path);
    return true;
}

bool changeIteration(caffe::SolverParameter &solver_param, int iterations) {
    if (iterations < 0) {
        LOG(ERROR) << "Iterations must be greater than zero!" << std::endl;
        return false;
    }
    solver_param.set_max_iter(iterations);
    return true;
}

bool changeRate(caffe::NetParameter* net_param, double rate) {
    if (rate < 0 || rate > 1) {
        LOG(ERROR) << "Rate must be in range of 0 to 1!" << std::endl;
        return false;
    }
    net_param->mutable_layer(33)->mutable_imp_map_param()->set_cmp_ratio(static_cast<float>(rate));
    std::cout << "after modified, the rate is " <<net_param->mutable_layer(33)->mutable_imp_map_param()->cmp_ratio() << std::endl;
    return true;
}

int main(int argc, char** argv) {
    caffe::GlobalInit(&argc, &argv);
    FLAGS_logtostderr = 1;
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("/home/snk/bin/caffe-master/examples/image_compression/model_rc/solver.prototxt", &solver_param);
    if (FLAGS_save_path.size() > 0) {
        if (changeSavePath(solver_param, FLAGS_save_path)) {
            LOG(INFO) << "Save trained model files into " << FLAGS_save_path << std::endl;
        }
    }
    if (FLAGS_iter > 0) {
        if (changeIteration(solver_param, FLAGS_iter)) {
            LOG(INFO) << "The Number of train iteration is set to " << FLAGS_iter << std::endl;
        }
    }
    if (FLAGS_rate > 0) {
        if (changeRate(solver_param.mutable_net_param(), FLAGS_rate)) {
            LOG(INFO) << "The compression penalized threshold rate is set to " << FLAGS_rate << std::endl;
        }
    }
    // std::cout << FLAGS_iter << " , " <<  FLAGS_rate << std::endl; 
    LOG(INFO) << "Train Start." << std::endl;
    shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    solver->Solve();
    LOG(INFO) << "Train Done." << std::endl;
    return 0;
}