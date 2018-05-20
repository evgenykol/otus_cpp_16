#pragma once

#include <dlib/matrix.h>
#include <string>
#include <vector>

const int crit_num = 7;

using sample_type = dlib::matrix<double,crit_num,1>;

void input_string_to_samples(std::string &line, std::vector<sample_type> &samples);
std::string sample_to_string(const sample_type &sample);
