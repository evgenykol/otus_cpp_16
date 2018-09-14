#include <iostream>
#include <vector>
#include <fstream>

#include <dlib/svm_threaded.h>
#include <dlib/clustering.h>
#include <dlib/rand.h>

#include "version.h"
#include "rlib.h"

using namespace dlib;

using linear_kernel_type = linear_kernel<sample_type>;

int main(int argc, char* argv[])
{
    try
    {
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

            if(nclusters < 3)
            {
                std::cout << "enter more clusters! \n";
                return 1;
            }

            std::cout << "rclst num clusters: " << nclusters << ", model file name: " << modelfname << std::endl;
        }
        else
        {
            std::cerr << "Usage: rclst <nclusters> <modelfname>\n";
            return 1;
        }

        //stdin data parsing
        std::string line;
        //freopen("dataset.csv", "rt", stdin);
        while(std::getline(std::cin, line))
        {
            input_string_to_samples(line, samples);
        }
        std::cout << "Parsing sucessfull!\n";



        //clusterization & saving clusters to files
        using ovo_trainer = one_vs_one_trainer<any_trainer<sample_type>>;
        ovo_trainer trainer;
        svr_linear_trainer<linear_kernel_type> linear_trainer;

        linear_kernel_type kernel;
        trainer.set_trainer(linear_trainer);

        pick_initial_centers(nclusters, initial_centers, samples, kernel);
        std::vector<unsigned long> assignments = spectral_cluster(kernel, samples, nclusters);
        std::cout << "Spectral cluster sucessfull! Assignments size = " << assignments.size()  <<
                  " samples number = " << samples.size() << std::endl;

        std::vector<std::ofstream> ofs;
        ofs.reserve(nclusters);
        for(int i = 0; i < nclusters; ++i)
        {
            ofs.push_back(std::ofstream(modelfname + "." + std::to_string(i)));
        }

        labels.reserve(samples.size());
        for(auto i = 0; i < samples.size(); ++i)
        {
            auto label = assignments[i];
            labels.push_back(label);
            ofs[label] << sample_to_string(samples[i]) << "\n";
        }

        for(auto &of : ofs)
        {
            of.close();
        }

        //file with number of clusters
        std::ofstream of_num(modelfname + ".num");
        of_num << nclusters << "\n";
        of_num.close();

        std::cout << "Clusterization sucsessful! ";
        std::cout << "(samples: " << samples.size() << ", lables: " << labels.size() << ", clusters: " << initial_centers.size() << ")\n";


        //decision function generating
        one_vs_one_decision_function<ovo_trainer, decision_function<linear_kernel_type>> df = trainer.train(samples, labels);

        std::cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << std::endl;
        std::cout << "predicted label: "<< df(samples[100])  << ", true label: "<< labels[100] << std::endl;
        std::cout << "predicted label: "<< df(samples[3333])  << ", true label: "<< labels[3333] << std::endl;
        std::cout << "predicted label: "<< df(samples[22222])  << ", true label: "<< labels[22222] << std::endl;
        std::cout << "Training sucsessful! \n";


//        //decision function serializing
//        one_vs_one_decision_function<ovo_trainer,
//        decision_function<linear_kernel_type>,  // This is the output of the poly_trainer
//        decision_function<linear_kernel_type>    // This is the output of the rbf_trainer
//        > df2;

//        df2 = df;
        serialize(modelfname + ".df") << df;
        std::cout << "Serializing sucsessful! \n";
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}


