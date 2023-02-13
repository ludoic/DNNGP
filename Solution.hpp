//
//  Solution.hpp
//  DNNGP_CLASSES
//
//  Created by Ludovico Italo Casati and Alexandre Pugin
//



#ifndef Solution_hpp
#define Solution_hpp

#include <iostream>
#include <fstream>
#include <array>
#include <string>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <vector>

#include <stdio.h>



class Chain_result {
    
public:
    size_t steps;
    size_t n;
    
    Chain_result(size_t n_steps, size_t n): steps(n_steps),n(n){
        sigma2.resize(steps);
        tau2.resize(steps);
        a.resize(steps);
        c.resize(steps);
        k.resize(steps);
        W.resize(steps);
        BETA.resize(steps);
        
        W[0].resize(n);
    };
    
    
    std::vector<Eigen::VectorXd> BETA;
    
    std::vector<double> sigma2;
    
    std::vector<double> tau2;
    
    std::vector<double> a;
    
    std::vector<double> c;
    
    std::vector<double> k;
    
    std::vector<Eigen::VectorXd> W;
    
    
    void save_to_file(std::string filename, bool short_=false){
    //exception handling
    try {
        std::cout << "Writing  array contents to file..."<<std::endl;
      //open file for writing
        std::ofstream fw(filename, std::ofstream::out);
        
      //check if file was successfully opened for writing
      if (fw.is_open())
      {
        //store array contents to text file
          size_t max_col=6+n;
          if(short_){max_col=6;}
          
        for (size_t i = 0; i <steps; i++) {
            fw << sigma2[i]<<","<< a[i]<<","<< c[i]<<","<< k[i]<<","<< tau2[i]<<","<<BETA[i](0)<<",";
            for(size_t j = 0; j < W[i].size(); j++){
                if(j!=W[i].size()-1)
                {fw << W[i](j)<<",";}
                else
                {fw << W[i](j);}
            }
            fw<<std::endl;
        }
        fw.close();
          
          std::cout << "Done writing"<<std::endl;
      }
      else std::cout << "Problem with opening file"<<std::endl;
    }
    catch (const char* msg) {
      std::cerr << msg << std::endl;
    }}

    
    
    
    void save_prediction(std::vector<double> & prediction, std::vector<std::array<double,3>> points, std::string filename, bool short_=false){
    //exception handling
    try {
        std::cout << "Writing  array contents to file..."<<std::endl;
      //open file for writing
        std::ofstream fw(filename, std::ofstream::out);
        
      //check if file was successfully opened for writing
      if (fw.is_open())
      {
        //store array contents to text file
          size_t max_col=6+n;
          if(short_){max_col=6;}
          
        for (size_t i = 0; i <prediction.size(); i++) {
            fw << points[i][0]<<","<< points[i][1]<<","<< points[i][2]<<","<< prediction[i]<<std::endl;
        }
        fw.close();
          
          std::cout << "Done writing"<<std::endl;
      }
      else std::cout << "Problem with opening file"<<std::endl;
    }
    catch (const char* msg) {
      std::cerr << msg << std::endl;
    }}

    
};

#endif /* Solution_hpp */
