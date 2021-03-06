#include "rlib.h"

#include <boost/algorithm/string.hpp>

void input_string_to_samples(std::string &line, std::vector<sample_type> &samples)
{
    sample_type m;
    std::vector<std::string> tokens;
    boost::trim(line);
    boost::split(tokens, line, boost::is_any_of(";"));

    for (int i = 0; i < crit_num - 1; ++i)
    {
        if(tokens.at(i).length() < 1)
        {
            tokens.at(i) = "0.0";
        }
        m(i) = stod(tokens.at(i));
    }
    if ((tokens.at(crit_num-1).length() < 1) || (tokens.at(crit_num).length() < 1))
    {
        m(crit_num-1) = 0;
    }
    else
    {
        m(crit_num-1) = (stoi(tokens.at(crit_num-1)) == stoi(tokens.at(crit_num)));
    }
    samples.push_back(m);
}

std::string sample_to_string(const sample_type &sample)
{
    std::string result;
    for(int i = 0; i < crit_num - 1; ++i)
    {
        result += std::to_string(sample(i)) + ";";
    }

    int flag = sample(crit_num-1);
    result += std::to_string(flag);
    return result;
}

bool string_to_sample(std::string &line, sample_type &sample)
{
    std::vector<std::string> tokens;
    boost::trim(line);
    boost::split(tokens, line, boost::is_any_of(";"));

    if(tokens.size() < crit_num - 1)
    {
        std::cout << "Bad input string format!\n";
        return false;
    }

    for (int i = 0; i < crit_num - 1; ++i)
    {
        if(tokens.at(i).length() < 1)
        {
            tokens.at(i) = "0.0";
        }
        sample(i) = stod(tokens.at(i));
    }
    return true;
}

