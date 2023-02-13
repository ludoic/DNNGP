//
//  Data_structure.hpp
//  DNNGP_CLASSES
//
//  Created by Ludovico Italo Casati and Alexandre Pugin
//







#ifndef Data_structure_hpp
#define Data_structure_hpp

#include <iostream>
#include <fstream>
#include <array>
#include <string>
#include <stdio.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>


template<unsigned Type, unsigned Size>
class SPdata {
    
    //SPdata(std::initializer_list<double> p){};
    
public:
    
    SPdata(size_t space_size, size_t time_size, size_t n_covariates):
    space_size(space_size), time_size(time_size), n_points (Size), n_x(n_covariates), space_distance(Eigen::MatrixXd::Zero(space_size, space_size)){

    };
    
    SPdata() = default;
    
    SPdata(size_t space_size, size_t time_size, size_t n_covariates, std::string file):
    space_size(space_size), time_size(time_size), n_points (space_size*time_size), n_x(n_covariates), space_distance(Eigen::MatrixXd::Zero(space_size, space_size)){
        x.resize(Size,n_covariates);
        read_file(file);
        compute_spd();
    };
    
    friend class Solver;
    
    
    std::array<std::array<double,Type>,Size> data_points;  // time, space
    
    Eigen::VectorXd y;
    
    Eigen::VectorXd w0;
    
    Eigen::MatrixXd x;
    size_t n_x;
    
    size_t space_size;
    size_t time_size;
    size_t n_points;
    
    
    Eigen::MatrixXd space_distance;
    
    double s_dist (const size_t p1, const size_t p2){
        
        auto pp1=data_points[p1];
        auto pp2=data_points[p2];
  
        return std::sqrt((pp1[1]-pp2[1])*(pp1[1]-pp2[1])+(pp1[2]-pp2[2])*(pp1[2]-pp2[2]));
        
    };
    
    double s_dist (const size_t p1, std::array<double,Type> p2){
        
        auto pp1=data_points[p1];
        auto pp2=p2;
  
        return std::sqrt((pp1[1]-pp2[1])*(pp1[1]-pp2[1])+(pp1[2]-pp2[2])*(pp1[2]-pp2[2]));
        
    };
    
    double t_dist (const size_t p1, const size_t p2){
        auto pp1=data_points[p1];
        auto pp2=data_points[p2];
        return std::abs((pp1[0]-pp2[0]));
    };
    
    double t_dist (const size_t p1, std::array<double,Type> p2){
        auto pp1=data_points[p1];
        auto pp2=p2;
        return std::abs((pp1[0]-pp2[0]));
    };
    
    double max_t(){
        double ret(0);
        for(auto &p:this->data_points){
            ret=std::max(ret,p[0]);
        }
        return ret;
    }
    
    double max_space(){
        double ret(0);
        for(size_t i=0;i<space_size;i++){

            auto dummy(this->space_distance.row(i));
            double dist(0);
            for(auto &d:dummy){
                dist=std::max(dist,d);
            }
            ret=std::max(ret,dist);
        }
        
        return ret;
    }
    
    double min_coor(){
        double ret(0);
        for(size_t i=0;i<n_points;i++){
            auto it=std::min(data_points[i][1],data_points[i][2]);
            ret=std::min(ret,it);
        }
        
        return ret;
    }
    
    size_t time_index(double time){
        for(size_t i=0;i<Size;i++){
            if(time<data_points[i][0] ){
                if(i==0){return 0;}
                return (size_t)((i/this->space_size)-1);
            }}
        
        return this->time_size;
        }
    
    
    size_t space_index(std::pair<double,double> location){
        
        for(size_t i=0;i<Size;i++){
            bool check(data_points[i][1]==location.first && data_points[i][2]==location.second);
            if(check){
                return (size_t)i%this->space_size;
            }}
        return this->space_size;
        
    }
    
    double mean (){
        double mean(0);
        for(auto &i:y)
            mean+=i;
        mean/=n_points;
        return mean;
    }
    
    double var (){
        double mean_(this->mean());
        double var(0);
        for(auto &i:y)
            var+=(i-mean_)*(i-mean_);
        var/=(n_points-1);
        return var;
    }

    
    void construct_grid(size_t time, size_t spacex, size_t spacey, double size_time,double size_space,double space_min){
            
        std::vector<double> x1,x2,t;
        
        x1.resize(spacex);
        x2.resize(spacey);
        t.resize(time);
            
            //generate locations
            for (size_t i=0; i<spacex;i++){
                x1[i]=size_space* i /spacex + space_min;
                
            }
            for (size_t i=0; i<spacey;i++){
                x2[i]=size_space*i/spacey + space_min;
                
            }
            for (size_t i=0; i<time;i++){
            t[i]=size_time*i/spacey;
            
            }
            
            // build points_ds
            size_t l(0);
        
            for(size_t i=0; i<time;i++){
                for (size_t j=0; j<spacex;j++) {
                    for(size_t k=0;k<spacey;k++){
                        
                    std::array<double,3> p{ x1[j], x1[k], t[i] };
                    this->data_points[l++]=p;
            }}}
        
        
        this->space_size=spacex*spacey;
        this->time_size=time;
        this->n_points=Size;
        


    }

    
    void set_covariates (size_t n_c, Eigen::MatrixXd covariates)
    {
        this-> x=covariates;
        this-> n_x=n_c;
    }
    
    Eigen::MatrixXd get_covariates ()
    {
        return this-> covariates;
    }
    
private :
    
    void compute_spd (){
            for(size_t i=0; i<space_size;i++){
                for(size_t j=i+1; j<space_size;j++){
                    double dist=s_dist(i,j);
                    space_distance(i,j)=dist;
                    space_distance(j,i)=dist;}}
    }
    
    void read_file(std::string filename, bool sinth=false){
        y.resize(n_points);
        x.resize(n_points,n_x);
        
        Eigen::MatrixXd data;
        
        double n_col(5+n_x);
        
        // time, space1, space2, space3, y, x1, x2, x3, ......
        // +1 since now we have w0 as well
        
        data.resize(n_points,n_col+1);
        
        std::ifstream myfile(filename);
        std::string line;
        if(myfile.is_open()){
            size_t row(0);
            while(std::getline(myfile, line)) {
        
            std::istringstream s(line);
            std::string field;
            size_t col(0);
    
            while (getline(s, field,',')){
                data(row,col)=std::stod(field);
                col++;
            }
                row++;
        }}
        else {std::cout << "Couldn't open file"<<std::endl;}
     
        
        // fill data structures
        y=data.col(3);
        
        if(sinth){
            w0=data.col(4);}
      
        for(size_t j=0; j<n_points; j++){
            data_points[j][0]=data(j,2);
            
            for(size_t i=0; i<2; i++){
                data_points[j][i+1]=data(j,i);
                }
            
            for(size_t k=0; k<n_x; k++){
                x(j,k)=data(j,1+k);
                }
        }
        
        
        
    }
    
    
    
};










#endif /* Data_structure_hpp */
