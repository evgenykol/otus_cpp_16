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
using ovo_trainer = one_vs_one_trainer<any_trainer<sample_type>>;
using df_type =
    one_vs_one_decision_function<ovo_trainer,
    decision_function<linear_kernel_type>,
    decision_function<linear_kernel_type>>;

void do_classification(const std::string &mfname_, int nclust_, df_type &df_, std::string &line_);
double earth_dist(double y1, double x1, double y2, double x2);

int main(int argc, char* argv[])
{
    try
    {
        //console input
        int nclusters;
        std::string modelfname;

        if ((argc > 1) &&
                (!strncmp(argv[1], "-v", 2) || !strncmp(argv[1], "--version", 9)))
        {
            std::cout << "version " << version() << std::endl;
            return 0;
        }
        else if (argc == 2)
        {
            modelfname = argv[1];
            std::cout << "rclss, model file name: " << modelfname << std::endl;
        }
        else
        {
            std::cerr << "Usage: rclss <modelfname>\n";
            return 1;
        }

        //reading number of clusters
        std::ifstream if_num(modelfname + ".num");
        if(if_num.is_open())
        {
            if_num >> nclusters;
            if_num.close();
        }
        else
        {
            std::cout << "number of clusters data file doesn't exist! \n";
            return 1;
        }

        //reading decision function
        df_type df;
        deserialize(modelfname + ".df") >> df;

        //read stdin and classify
        std::string line;
        //freopen("test.txt", "rt", stdin);
        while(std::getline(std::cin, line))
        {
            do_classification(modelfname, nclusters, df, line);
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}

void do_classification(const std::string &mfname_, int nclust_, df_type &df_, std::string &line_)
{
    sample_type sample;
    string_to_sample(line_, sample);

    unsigned long cluster = df_(sample);
    //std::cout << "cluster: " << cluster << " of " << nclust_ << std::endl;

    if(cluster > nclust_ - 1)
    {
        throw std::logic_error("classification error, no such cluster!");
    }

    //reading cluster data from file
    std::string clustfname = mfname_ + "." + std::to_string(cluster);
    std::ifstream ifs(clustfname);
    if(!ifs.is_open())
    {
        throw std::logic_error("cluster data file doesn't exist: " + clustfname);
    }

    std::vector<sample_type> cluster_samples;
    std::string lc;
    while(std::getline(ifs, lc))
    {
        sample_type sc;
        string_to_sample(lc, sc);
        cluster_samples.push_back(sc);
    }
    ifs.close();

    std::sort(cluster_samples.begin(), cluster_samples.end(),
              [&sample](const sample_type &l, const sample_type &r)
    {
        return earth_dist(sample(0), sample(1), l(0), l(1)) < earth_dist(sample(0), sample(1), r(0), r(1));
    }
            );

    //std::cout << "\nnearest flats:\n";
    for(auto &s : cluster_samples)
    {
        std::cout << sample_to_string(s) << "\t distance: " << earth_dist(sample(0), sample(1), s(0), s(1)) << "\n";
    }
}

double earth_dist(double y1, double x1, double y2, double x2)
{
    //https://en.wikipedia.org/wiki/Great-circle_distance
    using namespace std;

    const double grad_to_rad = 0.0174533; //convertion from degress to radians
    auto dx = (x2 - x1) * grad_to_rad;
    auto dy = (y2 - y1) * grad_to_rad;

    auto sq = pow(sin(dx/2), 2) + cos(x1*grad_to_rad) * cos(x2*grad_to_rad) * pow(sin(dy/2), 2);
    auto dd = 2 * asin(sqrt(sq));
    auto dist = dd * 6371; //Earth radius
    return dist;
}
