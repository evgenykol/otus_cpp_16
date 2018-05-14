#include <iostream>
#include <vector>
#include <fstream>

#include <dlib/svm_threaded.h>
#include <dlib/clustering.h>
#include <dlib/rand.h>

#include <boost/algorithm/string.hpp>

#include "version.h"

//using namespace std;
using namespace dlib;

constexpr int crit_num = 7;

using sample_type = matrix<double,crit_num,1>;
using kernel_type = linear_kernel<sample_type>;

int main(int argc, char* argv[])
{
    try
    {
        kcentroid<kernel_type> kc(kernel_type(), 0.01, 8);
        kkmeans<kernel_type> test(kc);

        std::vector<sample_type> samples;
        std::vector<sample_type> initial_centers;
        std::vector<double> labels;

        //console input
        int nclusters;
        std::string modelfname;

        if ((argc > 1) &&
                (!strncmp(argv[1], "-v", 2) || !strncmp(argv[1], "--version", 9)))
        {
            std::cout << "version " << version() << std::endl;
            return 0;
        }
        else if (argc == 3)
        {
            nclusters = atoi(argv[1]);
            modelfname = argv[2];
            std::cout << "rclst num clusters: " << nclusters << ", model file name: " << modelfname << std::endl;
        }
        else
        {
            std::cerr << "Usage: kkmeans <nclusters> <modelfname>\n";
            return 1;
        }

        //stdin data parsing
        sample_type m;
        std::string line;
        freopen("dataset.csv", "rt", stdin);
        while(std::getline(std::cin, line))
        {
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
            if ((tokens.at(6).length() < 1) || (tokens.at(7).length() < 1))
            {
                m(6) = 0;
            }
            else
            {
                m(6) = (stoi(tokens.at(6)) == stoi(tokens.at(7)));
            }
            samples.push_back(m);
        }

        std::cout << "Parsing sucessfull!\n";

        //clusterization
        test.set_number_of_centers(nclusters);
        pick_initial_centers(nclusters, initial_centers, samples, test.get_kernel());
        test.train(samples, initial_centers);

        //std::ofstream of(modelfname);

        for(auto &s : samples)
        {
            labels.reserve(samples.size());
            labels.push_back(test(s));
            //of << s(0) << "; " << s(1) << ";  " << test(s) << "\n";
        }
        //of.close();

        std::cout << "Clusterization sucsessful!  ";
        std::cout << "samples: " << samples.size() << ", clusters: " << initial_centers.size() << ", lables: " << labels.size() << "\n";

        //decision function generating
        using ovo_trainer = one_vs_one_trainer<any_trainer<sample_type>>;
        ovo_trainer trainer;

        krr_trainer<kernel_type> linear_trainer;
        linear_trainer.set_kernel(kernel_type());
        trainer.set_trainer(linear_trainer);

        one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

//        std::cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << std::endl;
//        std::cout << "predicted label: "<< df(samples[100])  << ", true label: "<< labels[100] << std::endl;
//        std::cout << "predicted label: "<< df(samples[3333])  << ", true label: "<< labels[3333] << std::endl;
//        std::cout << "predicted label: "<< df(samples[22222])  << ", true label: "<< labels[22222] << std::endl;
        std::cout << "Training sucsessful! \n";

        //decision function serializing
        auto ddf = df;
        serialize(modelfname + ".df") << df;
        std::cout << "Serializing sucsessful! \n";
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}


