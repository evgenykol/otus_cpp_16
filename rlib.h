#pragma once

#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>
#include <dlib/clustering.h>

#include <string>
#include <vector>

const int crit_num = 7;

using sample_type = dlib::matrix<double,crit_num-1,1>;

using linear_kernel_type = dlib::linear_kernel<sample_type>;
using ovo_trainer_type = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
using ovo_df_type = dlib::one_vs_one_decision_function<ovo_trainer_type, dlib::decision_function<linear_kernel_type>>;

void input_string_to_samples(std::string &line, std::vector<sample_type> &samples);
std::string sample_to_string(const sample_type &sample);
void string_to_sample(std::string &line, sample_type &sample);
