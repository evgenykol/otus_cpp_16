#include <iostream>
#include <vector>
#include <fstream>

#include <dlib/clustering.h>
#include <dlib/rand.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

#include <boost/algorithm/string.hpp>

#include "version.h"

using namespace std;
using namespace dlib;

constexpr int crit_num = 7;

int main(int argc, char* argv[])
{
    try
    {
        typedef matrix<double,crit_num,1> sample_type;
        typedef radial_basis_kernel<sample_type> kernel_type;

        kcentroid<kernel_type> kc(kernel_type(0.1),0.01, 8);
        kkmeans<kernel_type> test(kc);

        std::vector<sample_type> samples;
        std::vector<sample_type> initial_centers;

        int nclusters;
        string modelfname;

        if ((argc > 1) &&
                (!strncmp(argv[1], "-v", 2) || !strncmp(argv[1], "--version", 9)))
        {
            cout << "version " << version() << endl;
            return 0;
        }
        else if (argc == 3)
        {
            nclusters = atoi(argv[1]);
            modelfname = string(argv[2]);
            cout << "rclst num clusters: " << nclusters << ", model file name: " << modelfname << endl;
        }
        else
        {
            std::cerr << "Usage: kkmeans <nclusters> <modelfname>\n";
            return 1;
        }

        sample_type m;
        string line;
        freopen("dataset.csv", "rt", stdin);
        while(getline(cin, line))
        {
            std::vector<string> tokens;
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

        cout << "parse sucessfull! \n";

//        test.set_number_of_centers(n);
//        pick_initial_centers(n, initial_centers, samples, test.get_kernel());
//        test.train(samples,initial_centers);

//        ofstream of("kkmeans_ex_out.txt");

//        array2d<rgb_pixel> img;
//        img.set_size(200, 200);

//        for (auto &pix : img)
//        {
//            pix = rgb_pixel(255, 255, 255);
//        }

//        for(auto &s : samples)
//        {
//            auto x = s(0) + 100;
//            auto y = s(1) + 100;
//            auto c = test(s) + 1;
//            img[x][y] = colormap_jet(c, 0, n);

//            of << s(0) << ";" << s(1) << ";" << c <<"\n";
//        }
//        of.close();
//        save_bmp(img, "./kkmeans.bmp");
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}


