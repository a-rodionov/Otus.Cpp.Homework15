#include <iostream>
#include <vector>

#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace dlib;

// Here we declare that our samples will be 2 dimensional column vectors.  
// (Note that if you don't know the dimensionality of your vectors at compile time
// you can change the 2 to a 0 and then set the size at runtime)
typedef matrix<double,2,1> sample_type;

// Now we are making a typedef for the kind of kernel we want to use.  I picked the
// radial basis kernel because it only has one parameter and generally gives good
// results without much fiddling.
typedef radial_basis_kernel<sample_type> kernel_type;

// This ctype facet classifies commas and endlines as whitespace
struct csv_whitespace : std::ctype<char> {
  static const mask* make_table()
  {
    // make a copy of the "C" locale table
    static std::vector<mask> v(classic_table(), classic_table() + table_size);
    v[';'] |=  space;  // ; will be classified as whitespace
    v[' '] &= ~space;      // space will not be classified as whitespace
    return &v[0];
  }
  csv_whitespace(std::size_t refs = 0) : ctype(make_table(), false, refs) {}
};

int main(int argc, char const* argv[])
{
  try
  {
    unsigned long long cluster_num;
    try
    {
      if(2 != argc) {
        throw std::invalid_argument("");
      }

      std::string digit_str{argv[1]};
      if(!std::all_of(std::cbegin(digit_str),
                      std::cend(digit_str),
                      [](unsigned char symbol) { return std::isdigit(symbol); } )) {
        throw std::invalid_argument("");
      }
      cluster_num = std::stoull(digit_str);
      if(0 == cluster_num) {
        throw std::invalid_argument("");
      }
    }
    catch(...)
    {
      std::string error_msg = "The programm must be started with 1 parameter, which means the cluster number. "
                              "The value must be in range 1 - "
                              + std::to_string(std::numeric_limits<unsigned long long>::max());
      throw std::invalid_argument(error_msg);
    }

    std::vector<sample_type> samples;

    std::locale locale_(std::locale(""), new csv_whitespace);
    std::string line;
    while( std::getline(std::cin, line) )
    {
      std::istringstream iss(line);
      iss.imbue(locale_);
      sample_type m;
      iss >> m(0) >> m(1);
      samples.push_back(std::move(m));
    }

    // Here we declare an instance of the kcentroid object.  It is the object used to 
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters 
    // you need to set.  The first argument to the constructor is the kernel we wish to 
    // use.  The second is a parameter that determines the numerical accuracy with which 
    // the object will perform part of the learning algorithm.  Generally, smaller values 
    // give better results but cause the algorithm to attempt to use more dictionary vectors 
    // (and thus run slower and use more memory).  The third argument, however, is the 
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.  
    kcentroid<kernel_type> kc(kernel_type(0.00001), 0.01, 8);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    kkmeans<kernel_type> kkmeansObj(kc);

    // tell the kkmeans object we made that we want to run k-means with k set to cluster_num. 
    // (i.e. we want cluster_num clusters)
    kkmeansObj.set_number_of_centers(cluster_num);

    std::vector<sample_type> initial_centers;

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    pick_initial_centers(cluster_num, initial_centers, samples, kkmeansObj.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    kkmeansObj.train(samples, initial_centers);

    // now loop over all our samples and print out their predicted class.  In this example
    // all points are correctly identified.
    for(const auto& data : samples) {
      std::cout << data(0) << ';' << data(1) << ';' << kkmeansObj(data) << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }
  return 0;

}
