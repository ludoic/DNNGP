//
//  Solver.hpp
//  DNNGP_CLASSES
//
//  Created by Ludovico Italo Casati and Alexandre Pugin
//



#ifndef Solver_hpp
#define Solver_hpp

#include "Data_structure.hpp"
#include "Solution.hpp"

#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <vector>
#include <unordered_map>
#include <string>

#include <random>

#include <chrono>

#include <cmath>



template<unsigned Type, unsigned Size>
class DNNGP {

public:
    
    DNNGP(size_t space_size, size_t time_size, size_t n_covariates,std::string filename):
    
    data(space_size, time_size,n_covariates, filename)
    
    {
        n=data.n_points;
        E.resize(n);
        N.resize(n);
        U.resize(n);
        A.resize(n);
        f.resize(n);
        C.resize(n);
        Cn.resize(n);
        w0.resize(n);
        
        COV.resize(n,n);
        COV_new.resize(n,n);
        n_covar=n_covariates;
        
        b={};
    };
    
    DNNGP(SPdata<Type,Size> datas):data(datas)
    {
        n=data.n_points;
        E.resize(n);
        N.resize(n);
        U.resize(n);
        A.resize(n);
        f.resize(n);
        C.resize(n);
        Cn.resize(n);
        w0.resize(n);
        
        COV.resize(n,n);
        COV_new.resize(n,n);
        n_covar=data.n_x;
        
        b={};
    };
    
private:
    
    // structures
    SPdata<Type,Size> data;
    
    size_t n;
    
    size_t n_covar;
    
    Eigen::VectorXd w0;
    
    Eigen::MatrixXd X;
    
    std::vector<std::vector<size_t>> E;
    
    std::vector<std::vector<size_t>> N;
    
    std::vector<std::vector<size_t>> U;
    
    std::vector<Eigen::VectorXd> A; //Anl container
    
    std::vector<Eigen::VectorXd> C; //Cnl container
    
    std::vector<Eigen::MatrixXd> Cn; //Cnn container
    
    std::vector<double> f; // f_l container
    
    std::unordered_map<size_t, double*> b; // b_l_l container
    
    size_t m; // number of neighbors
    
    Eigen::MatrixXd COV;
    
    Eigen::MatrixXd COV_new;
    
    // functions
    
    double cov_matern (double& h, double& u, double& sigma2, double& a, double& c, double& k){
        double ret(0);
        double term(a*u*u+1);
        ret= ( sigma2/std::pow(term,k) ) * std::exp( (-c*h) / (std::pow(term,k*0.5)));
        return ret;
    }
    
    
    void set_neighbors(size_t mm){
        m=mm;
    }
    
    
    std::vector<size_t> compute_E_l (const size_t l){
        
        std::vector<size_t> res; //output, right now is a vector of size_t
    
        size_t t_indxu = (size_t)l/data.space_size;    //index of time for point l, it is also the current block
        size_t s_indxu = l%data.space_size;         //index of space for point l

        // sort a vector of indexes according to distance matrix
        Eigen::VectorXd distances= data.space_distance.row(s_indxu);
        std::vector<size_t> index(distances.size(), 0);
        for (size_t i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }
        
        sort(index.begin(), index.end(),
             [&](const size_t& a, const size_t& b){
            return (distances[a] < distances[b]);});
        // now index={index of closest point, index of second closest point, ... , index of furthest point}
        // of course indx[0]=myself
        
        // for all points at previous times, get the index of the m/k closest point
            for(size_t k=1; k<=std::min(t_indxu,m); k++){
                size_t mk = (size_t)m/k;
                // start at i=1 since one of the distance values will be 0
                for(size_t i=0; i<mk && i<data.space_size; i++){
                    res.push_back(index[i]+(t_indxu-k)*data.space_size);
                }
            }
        
          size_t i=1;
          size_t count=0;
          while(count<std::min(m,s_indxu) && i<index.size()){
              if(index[i]<s_indxu){
                  count++;
                  res.push_back(index[i]+(t_indxu)*data.space_size);}
              i++;
          }
        return res;
    }
    
    
    std::vector<size_t> compute_E_pred (const Eigen::VectorXd space_distance, size_t time, size_t space){
        
        
        std::vector<size_t> res; //output, right now is a vector of size_t
    
      
        size_t t_indxu = time;    //index of time for point l, it is also the current block
        size_t s_indxu = space;         //index of space for point l

        // sort a vector of indexes according to distance matrix
        Eigen::VectorXd distances=space_distance;
        
        std::vector<size_t> index(distances.size(), 0);
        for (size_t i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }
        
        sort(index.begin(), index.end(),
             [&](const size_t& a, const size_t& b){
            return (distances[a] < distances[b]);});
        // now index={index of closest point, index of second closest point, ... , index of furthest point}
        // of course indx[0]=myself
        
        // for all points at previous times, get the index of the m/k closest point
            for(size_t k=1; k<=std::min(t_indxu,m); k++){
                size_t mk = (size_t)m/k;
                // start at i=1 since one of the distance values will be 0
                for(size_t i=0; i<mk && i<data.space_size; i++){
                    res.push_back(index[i]+(t_indxu-k)*data.space_size);
                    }
            }
        
        
        if(time!=data.time_size){
          size_t i=1;
          size_t count=0;
          while(count<std::min(m,s_indxu) && i<index.size()){
              if(index[i]<s_indxu){
                  count++;
                  res.push_back(index[i]+(t_indxu)*data.space_size);}
              i++;
              
          }}
        
        return res;
    }
    
    
    
    
    void compute_E(){
        for(size_t i=0; i<n;i++) {E[i]=compute_E_l(i);
        }
    }
    
    
    Eigen::MatrixXd update_COV (double& sigma2, double& a, double& c, double& k){
        Eigen::MatrixXd cov;    cov.resize(n,n);
        for(size_t i=0; i<n;i++){
            for(size_t j=i; j<n;j++){
                double s=data.s_dist(i,j);
                double t=data.t_dist(i,j);
                double value = cov_matern( s, t, sigma2, a, c, k);
                cov(i,j)=value;
                cov(j,i)=value;
            }}
        return cov;
    }
    
    Eigen::MatrixXd update_COV_pred (double& sigma2, double& a, double& c, double& k, std::array<double,3>point){
        Eigen::MatrixXd cov;    cov.resize(n+1,n+1);
        for(size_t i=0; i<n+1;i++){
            for(size_t j=i; j<n+1;j++){
                
                if(i==n || j==n){
                    double s=data.s_dist(i,point);
                    double t=data.t_dist(i,point);
                    double value = cov_matern( s, t, sigma2, a, c, k);
                    cov(i,j)=value;
                    cov(j,i)=value;}
                else{
                    double s=data.s_dist(i,j);
                    double t=data.t_dist(i,j);
                    double value = cov_matern( s, t, sigma2, a, c, k);
                    cov(i,j)=value;
                    cov(j,i)=value;}
            }}
        return cov;
    }
    
    
    
    std::vector<size_t> compute_N_l (const size_t l) {
        std::vector<size_t> Nl;
        std::vector<double> cov;
        std::vector<size_t> index;
        
        for (size_t i=0; i<E[l].size(); i++){
            cov.push_back(COV(l,E[l][i]));
            index.push_back(i);
        }
        
        std::sort(index.rbegin(), index.rend(),
             [&](const size_t& a, const size_t& b){
            return (cov[a] < cov[b]);});
        
        for(size_t j=0;(j<E[l].size() &&j<m);j++){
            Nl.push_back(E[l][index[j]]);
        }
        
        return Nl;
    }
    
    
    std::vector<size_t> compute_N_pred (const size_t l, Eigen::MatrixXd CVV, std::vector<size_t> Ee) {
        std::vector<size_t> Nl;
        std::vector<double> cov;
        std::vector<size_t> index;
        
        auto cvl=CVV.row(l);
         

        for (size_t i=0; i<Ee.size(); i++){
            cov.push_back(cvl(Ee[i]));
            index.push_back(i);
        }
        
        std::sort(index.rbegin(), index.rend(),
             [&](const size_t& a, const size_t& b){
            return (cov[a] < cov[b]);});
        
        for(size_t j=0;(j<Ee.size() &&j<m);j++){
            Nl.push_back(Ee[index[j]]);
        }
        
        return Nl;
    }

    
    void update_N(){
        N.clear();
        N.resize(n);
        for(size_t i=0; i<n;i++) {N[i]=compute_N_l(i);}
    }
    
    
    void update_U (){
        U.clear();
        U.resize(n);
        for(size_t i=0; i<N.size(); i++){
            for(size_t j=0; j<N[i].size(); j++){
                U[N[i][j]].push_back(i);
            }}
    }

    
    //  -------------------------------------- BETA:  --------------------------------------

    // V0_inv is already the inverted prior covariance for beta, just do it once at beginning
    // Vmu0 is the V0_inv*mu0 therm, just do it once at beginning
    Eigen::VectorXd sample_beta (const Eigen::MatrixXd& V0_inv, const Eigen::VectorXd& Vmu0, const double tau, Eigen::VectorXd& w, bool verbose=false){
        
        /*
        //    1D case
        double tau1=1/(tau*tau);
        Eigen::MatrixXd Xt(X.transpose());
        auto XtX=Xt*X;
        Eigen::MatrixXd V(V0_inv+tau1*XtX); // V=V_beta^(-1)
        Eigen::VectorXd mu(Vmu0);

        mu+= (Xt*(data.y-w)) * tau1; // mu

        // sample from normal(0,1)
        std::random_device rd;
        std::mt19937 eng(rd());

        //std::default_random_engine eng{1};
        std::normal_distribution<> dis{mu(0)/V(0,0),std::sqrt(1/V(0,0))};

        Eigen::VectorXd beta;
        beta.resize(1);
        beta(0) = dis(eng);
         */
        
        double tau1=1/(tau*tau);
        Eigen::MatrixXd Xt(X.transpose());
        auto XtX=Xt*X;
        Eigen::MatrixXd V(V0_inv+tau1*XtX); // V=V_beta^(-1)
        Eigen::VectorXd mu(Vmu0);
        
        mu+= (Xt*(data.y-w)) * tau1; // mu

        // sample from normal(0,1)
        std::random_device rd;
        std::mt19937 eng(rd());
        
        //std::default_random_engine eng{1};
        std::normal_distribution<> dis{0,1};
        
        auto p=Vmu0.size();
        Eigen::VectorXd rnormal;
        rnormal.resize(p);
        for(size_t i=0; i<p; i++){rnormal[i]=dis(eng);}
        
        Eigen::FullPivLU<Eigen::MatrixXd> lu(V);
        
        
        Eigen::MatrixXd L(V.llt().matrixL()); //cholesky decomp
        Eigen::FullPivLU<Eigen::MatrixXd> LU(L);

        
        Eigen::VectorXd beta(lu.solve(mu)+LU.solve(rnormal));
        
        if(verbose){
        std::cout<<std::endl;
        std::cout<<"Sampling beta:"<<std::endl;
            std::cout<<"value: "<<beta<<"; mean of posterior: "<<mu(0)/V(0,0)<<" variance of posterior: "<<std::sqrt(1/V(0,0))<<std::endl;}
         
        
        return beta;
    }
    
    
    //  -------------------------------------- TAU:  --------------------------------------

    double inv_gamma_sample (double alpha, double beta){
        std::random_device rd;
        std::mt19937 eng(rd());
        //c++ use scale parameter, so I need to sample with 1/beta since beta=rate
        std::gamma_distribution<> sample(alpha,(1/beta));
        // sample ~ gamma(alpha,b) -> 1/sample ~ inv_gamma(alpha,b)
        return (1/sample(eng));
    }

    // alpha=alpha0*n0/2 computed at the beginning
    double sample_tau (double alpha, double beta0, Eigen::VectorXd& w, Eigen::VectorXd& beta, bool verbose=false){
        
        Eigen::VectorXd ywbeta(data.y-X*beta-w);
        double b(beta0+0.5*(ywbeta.dot(ywbeta)));
        
        double sample(inv_gamma_sample(alpha,b)); // tau2 ~ inv_gamma(alpha,beta)
        
        if(verbose){
        std::cout<<std::endl;
        std::cout<<"Sampling tau:"<<std::endl;
        std::cout<<"value: "<<std::sqrt(sample)<<"; alpha_tau: "<<alpha<<" beta_tau: "<<b<<std::endl;}
        
        return std::sqrt(sample); //tau=sqrt(tau2)
        
    };

    
    //  -------------------------------------- Cnn, Cnl, Anl:  --------------------------------------
    
    
    Eigen::MatrixXd Cnn (const size_t &l, const Eigen::MatrixXd &cov){
        auto nn=N[l].size();
        Eigen::MatrixXd output = Eigen::MatrixXd::Zero(nn,nn);
        if(nn==0){
            return Eigen::MatrixXd::Zero(1,1);
        } //for the case where l has an empty neighbor set
        
        for(size_t i=0; i<nn;i++){
            for(size_t j=i;j<nn;j++){
                output(i,j)=cov(N[l][i],N[l][j]);
                output(j,i)=cov(N[l][i],N[l][j]);
            }}
        return output;
    }

      // return Cnl of point l
    Eigen::VectorXd Cnl (const size_t &l, const Eigen::MatrixXd &cov){
        auto nn=N[l].size();
        Eigen::VectorXd output;
        output.resize(nn);
        for(size_t i=0; i<nn;i++){
            output[i]=cov(l,N[l][i]);
        }
        return output;
    }
    
    // return a_nl of point l
    Eigen::VectorXd Anl (const Eigen::VectorXd & Cnl, const Eigen::MatrixXd & Cnn, bool verbose=false){
        if(Cnl.size()==0){
            return Eigen::VectorXd(0);
        }
        Eigen::FullPivLU<Eigen::MatrixXd> lu(Cnn);
        Eigen::VectorXd ret = lu.solve(Cnl);
        
        if(verbose){
            if(ret.norm()>100){std::cout <<std::endl;std::cout << "Norm Anl: " << ret.norm() << std::endl;std::cout << "Determinant Cnn: " << Cnn.determinant() << std::endl;std::cout <<std::endl;}}
        
        return ret;
    }
    
    
    
    Eigen::MatrixXd Cnn_pred (std::vector<size_t> Nnl, const Eigen::MatrixXd &cov){
        auto nn=Nnl.size();
        Eigen::MatrixXd output = Eigen::MatrixXd::Zero(nn,nn);
        if(nn==0){
            return Eigen::MatrixXd::Zero(1,1);
        } //for the case where l has an empty neighbor set
        
        for(size_t i=0; i<nn;i++){
            for(size_t j=i;j<nn;j++){
                output(i,j)=cov(Nnl[i],Nnl[j]);
                output(j,i)=cov(Nnl[i],Nnl[j]);
            }}
        return output;
    }

      // return Cnl of point l
    Eigen::VectorXd Cnl_pred (const size_t l, std::vector<size_t> Nnl, const Eigen::MatrixXd &cov){
        auto nn=Nnl.size();
        Eigen::VectorXd output;
        output.resize(nn);
        for(size_t i=0; i<nn;i++){
            output[i]=cov(l,Nnl[i]);
        }
        return output;
    }
    
    
    // f_l
    double f_l (const size_t &l, const Eigen::VectorXd & anl , const Eigen::MatrixXd &cov ,const Eigen::VectorXd & cnl){
       return cov(l,l)-cnl.dot(anl);
    }
    
    std::vector<double> compute_f (std::vector<Eigen::VectorXd> &full_anl,std::vector<Eigen::VectorXd>&full_cnl,const Eigen::MatrixXd &cov){
        std::vector<double> ret; ret.resize(n);
        for(size_t i=0; i<n;i++){
            ret[i]= f_l(i, full_anl[i], cov, full_cnl[i]);
        }
        return ret;
    }
    
    std::vector<Eigen::VectorXd> compute_Cnl (const Eigen::MatrixXd &cov)
    {
        std::vector<Eigen::VectorXd> ret; ret.resize(n);
        for(size_t i=0; i<n;i++){
            ret[i]= Cnl(i,cov);
        }
        return ret;
    }
    
    std::vector<Eigen::MatrixXd> compute_Cnn (const Eigen::MatrixXd &cov)
    {
        std::vector<Eigen::MatrixXd> ret; ret.resize(n);
        for(size_t i=0; i<n;i++){
            ret[i]= Cnn(i,cov);
        }
        return ret;
    }
    
    std::vector<Eigen::VectorXd> compute_A (const Eigen::MatrixXd &cov, std::vector<Eigen::VectorXd> &full_cnl, std::vector<Eigen::MatrixXd> &full_cnn){
        
        std::vector<Eigen::VectorXd> ret; ret.resize(n);
        
        for(size_t i=0; i<n;i++) {
            ret[i]=Anl(full_cnl[i],full_cnn[i]);
        }
        return ret;
    }
    
    
    //  -------------------------------------- W sampling:  --------------------------------------
    
    inline size_t key(size_t i, size_t j) {
        return (size_t) ((i+j)*(i+j+1))/2 + j; // Cantor's bijection from N² to N
    }
    
    // return b_l2_l1 for l1 neighbor of l2
    double* b_l2_l1(const size_t &l2, const size_t &l1, std::vector<Eigen::VectorXd> & full_Anl){
        
        size_t index = std::find(N[l2].begin(), N[l2].end(), l1) - N[l2].begin();
        if(index==N[l2].size()){
            //std::cout << "Error in b_l2_l1: " << l1 << " is not neighbor of " << l2 << std::endl;
            double NA = std::nan("");
            auto na_ptr = new double;
            *na_ptr = NA;
            return na_ptr;
        }
        auto ptr = new double;
        *ptr = full_Anl[l2](index);
        return ptr;
    }
    
    
    // update all b_l_l' needed to sample w_R
    void update_b(std::vector<Eigen::VectorXd> &full_Anl){
        b.clear();
        b = {};
        for(size_t l2=0; l2<n; l2++){
            //std::cout << "Computing for neighbors of " << l2 << std::endl;
            for(auto &l1 : N[l2]){
                b[key(l2,l1)] = b_l2_l1(l2,l1,full_Anl);
            }}
    }

    
    // return a_l2_l1 for l1 neighbor of l2
    double a_l2_l1(const size_t &l2,
                   const size_t &l1,
                   const std::vector<Eigen::VectorXd> &full_Anl,
                   const Eigen::VectorXd &w){
        double res = w(l2);
        for(auto &l : N[l2]){
            if(l != l1)
                res -= w(l) * *b.at(key(l2, l));
        }
        return res;
    }

    // return w_N(l)
    Eigen::VectorXd w_N_l(const size_t &l, const Eigen::VectorXd &w){
        auto n = N[l].size();
        Eigen::VectorXd res;
        res.resize(n);
        for(size_t i=0; i<n; i++){
            res(i) = w(N[l][i]);
        }
        return res;
    }
    
    Eigen::VectorXd w_N_l(std::vector<size_t> Nnl,const Eigen::VectorXd &w){
        auto n = Nnl.size();
        Eigen::VectorXd res;
        res.resize(n);
        for(size_t i=0; i<n; i++){
            res(i) = w(Nnl[i]);
        }
        return res;
    }

    double sample_w_r_l_i(const size_t &l_i,
                          const std::vector<Eigen::VectorXd> &full_Anl,
                          const std::vector<double> &full_f_l,
                          const Eigen::VectorXd &beta,
                          const double &tau,
                          const Eigen::VectorXd &w,
                          const Eigen::MatrixXd &cov){

        Eigen::VectorXd w_N_l_i = w_N_l(l_i, w);
        
        double fli( full_f_l[l_i] );

        double nu_l_i = 1/(tau*tau) + 1/fli;
        double inter_term =0;
        if(l_i!=0){
            inter_term = (full_Anl[l_i].dot(w_N_l_i));
        }
        
        double mu_l_i = (data.y(l_i) - X.row(l_i).dot(beta)) / (tau*tau) + inter_term/fli;
        
        for(auto &l : U[l_i]){
           
            const double &b_l_l_i = *b.at(key(l, l_i));
            
            double fl( full_f_l[l] );
            
            nu_l_i += b_l_l_i * b_l_l_i / fl;
            
            mu_l_i += b_l_l_i * a_l2_l1(l, l_i, full_Anl, w) / fl;
        }

        nu_l_i = 1 / nu_l_i;

        std::random_device rd;
        std::mt19937 eng(rd());

        std::normal_distribution<> sample_w{nu_l_i*mu_l_i, std::sqrt(nu_l_i)};

        return sample_w(eng);
    }
    
    
    Eigen::VectorXd sample_w_r(const std::vector<Eigen::VectorXd> &full_Anl,
                               const std::vector<double> &full_f_l,
                               const Eigen::VectorXd &beta,
                               const double &tau,
                               const Eigen::VectorXd &w){
        
        Eigen::VectorXd new_w(w);
        for(size_t l_i = 0; l_i<n; l_i++){
            new_w(l_i) = sample_w_r_l_i(l_i, full_Anl, full_f_l, beta, tau, new_w, COV);
        }
        
        return new_w;
    }

    //  -------------------------------------- Metropolis:  --------------------------------------
   
    double IG_log (double &sigma2, double& alpha, double& beta){
        double bs=beta/sigma2;
        //double bs=beta/sigma2;
        return -(alpha+1)*std::log( sigma2 ) - bs;
    }
    
    double logU_log(double &value, double max, double min){
        double ret(std::nan(""));
        if(min<=value && value<=max){
            ret= - value - std::log((max-min));
        }
        return ret;
    }
    
    double U_log(double &value, double max, double min){
        double ret(std::nan(""));
        if(min<=value && value<=max){
            ret=-std::log((max-min));}
        return ret;
        
    }
    
    double w_log(const std::vector<Eigen::VectorXd>& full_Anl,
                 const std::vector<double>& full_f_l,
                 Eigen::VectorXd& w){
        double ret(0);
        for(size_t i=0; i<n;i++){
            auto wNl( w_N_l(i,w) );
            double mean(full_Anl[i].dot(wNl));
            double fi(full_f_l[i]);
            double value = - 0.5 * std::log(fi) - (0.5 * (std::pow(w[i]-mean,2)) / fi);
            ret+=value;
            };
        return ret;
    }
     
    double step_U(double stepsize, double max, double min){
        return stepsize*(max-min);
    }

    double target_log (double sigma2, double& alpha, double& beta,
                       double a, double& max_a, double& min_a,
                       double c, double& max_c, double& min_c,
                       double k,
                       const std::vector<Eigen::VectorXd>& full_Anl,
                       const std::vector<double>& full_f_l,
                       Eigen::VectorXd& w,
                       bool verbose=false) {
        
        double ret(0);
        
        ret+=IG_log(sigma2, alpha, beta);
        ret+=logU_log(a,max_a,min_a);
        ret+=logU_log(c,max_c,min_c);
        ret+=logU_log(k,1.0,0.0);
        ret+=w_log(full_Anl,full_f_l,w);
        
        if(verbose){
            std::cout << "sigma2_log: "<<IG_log(sigma2, alpha, beta)<<std::endl;
            std::cout << "a_log: "<<logU_log(a,max_a,min_a)<<std::endl;
            std::cout << "c_log: "<<logU_log(c,max_c,min_c)<<std::endl;
            std::cout << "k_log: "<<logU_log(k,1.0,0.0)<<std::endl;
        }
        if(verbose){std::cout<<std::endl; std::cout << "log target: "<<ret <<std::endl;}

        return ret;
    }

    
    // Metropolis step
    std::pair<std::array<double, 4>,bool> Metropolis_step(
                              
                              double& sigma2, double& a, double& c, double& k,
                            
                                                          size_t n_parameters,
                              double stepsize,
                                                          
                              double alpha, double beta,
                              double max_a, double min_a,
                              double max_c, double min_c,
                                                          
                              Eigen::VectorXd& w,
                              bool verbose=false){

        
        Eigen::VectorXd thetak{{sigma2,a,c,k}};
        Eigen::VectorXd thetak1{{0,0,0,0}};

        std::random_device rd;
        std::random_device rd1;
        std::mt19937 eng(rd());
        std::mt19937 eng1(rd1());
        
        std::normal_distribution<> dis{0,1};
        std::uniform_real_distribution<> dis1{-1,1};
        
        Eigen::VectorXd step;
        step.resize(n_parameters);
        
        
        step(0)=stepsize*3*                           dis(eng);
        step(1)=step_U(stepsize, max_a, min_a)*     dis1(eng1);
        step(2)=step_U(stepsize, max_c, min_c)*     dis1(eng1);
        step(3)=step_U(stepsize, 1.0, 0.0)*         dis1(eng1);
        
        thetak1 = thetak + step;
        
        if(verbose){std::cout<<std::endl;
            std::cout<<"trial sigma: "<<thetak1[0]<<std::endl;
            std::cout<<"trial a: "<<thetak1[1]<<std::endl;
            std::cout<<"trial c: "<<thetak1[2]<<std::endl;
            std::cout<<"trial k: "<<thetak1[3]<<std::endl;
        }

        double sigma21=thetak1[0];
        double a1=thetak1[1];
        double c1=thetak1[2];
        double k1=thetak1[3];
        
        COV_new=update_COV (sigma21, a1, c1, k1);
        
        auto full_cnl_new=compute_Cnl(COV_new);
        auto full_cnn_new=compute_Cnn(COV_new);
        auto full_Anl_new=compute_A (COV_new,full_cnl_new,full_cnn_new);
        auto full_fl_new=compute_f(full_Anl_new, full_cnl_new, COV_new);
        
        
        double density_num=target_log (sigma21, alpha, beta,
                                       a1, max_a, min_a,
                                       c1, max_c, min_c,
                                       k1,
                                       full_Anl_new, full_fl_new,
                                       w,
                                       verbose);
        
        double density_den=target_log (sigma2, alpha, beta,
                                       a1, max_a, min_a,
                                       c1, max_c, min_c,
                                       k1,
                                       A, f,
                                       w,
                                       verbose);
        
        double fraq=density_num-density_den;
        fraq=std::exp(fraq);
        
        double alpha_ma(0);
        if(!isnan(fraq)){
            alpha_ma=std::min((double)1,fraq);}
        
        std::random_device rd_unif;
        std::mt19937 eng_unif(rd_unif());
        
        std::uniform_real_distribution<> dis_unif(0,1);
        double u_ma=dis_unif(eng_unif);
        
        if(u_ma<=alpha_ma)
        {
        // theta updated
            std::array<double,4> ret({thetak1(0),thetak1(1),thetak1(2),thetak1(3)});
            return std::make_pair(ret,true);
        }
        // theta not updated
        std::array<double,4> ret({thetak(0),thetak(1),thetak(2),thetak(3)});
        return std::make_pair(ret,false);
    }

    // Full MATERN COVARIANCE FUNCTION
    // the following require c++17
    /*
    double bessel_first(double nu,double x){
        if( ( (unsigned)(nu*2)) % 2 == 1 ){ // è un semi-intero
            return std::sph_bessel(nu,x);}
        else{ // è un intero
            return std::cyl_bessel_j(nu,x);}
        
    }
    
    double bessel_second(double nu,double x){
        double J(bessel_first(nu, x));
        double J_(bessel_first(-nu, x));
        return (std::cos(nu*M_PI)*J-J_)/std::sin(nu*M_PI);
        
    }
    
    double cov_matern (double& h, double& u, double& sigma2, double& a, double& c, double& k, double nu=3/2, double alpha=1){
        
        double ret(0);
        
        double time(a*std::pow(u,2*alpha)+1);
        double space(c*h);
        
        double time_k(std::pow(time,k/2));
        double gamma(std::tgamma(nu));
        
        double fraq(1/(gamma*time*std::pow(2,nu-1)));
                    
        double fraq_nu(std::pow(space/time_k,nu));
        
        double bessell(bessel_second(nu, space/time_k));
                    
        ret= sigma2 * fraq * fraq_nu * bessell;
        
        return ret;
    }*/
    

    
//  -------------------------------------- MCMC:  --------------------------------------
public:
    
    void initialize_w(bool zeros=false){
        if(zeros){
            for(size_t i=0; i<n;i++){w0(i)=0;}}
        else{
            w0=data.w0;
        }
    }
    
    void initialize_w(double value){
        for(size_t i=0; i<n;i++){w0(i)=data.y(i)-value;}
       
    }

    void set_X(Eigen::MatrixXd &x){
        X=x;
    }
    
    Chain_result run_MCMC (size_t n_steps,
                    size_t n_MALA_steps,
                    size_t n_neighbor,
                    
                    double& sigma2, double& a, double& c, double& k,
                    
                    double alpha_tau0, double beta_tau0,
                           
                    Eigen::VectorXd& Vmu0_beta,
                    Eigen::MatrixXd& V0_inv_beta,
                           
                    double alpha_sigma2, double beta_sigma2,
                           
                    double max_a, double min_a,
                    double max_c, double min_c,
                    
                    double tau_0,
                           
                    size_t n_metropolis_parameters, double stepsize_metropolis,
                    bool verbose=false, bool verbose_state=true){
        
        
        
        // ------------ initialize -----------
        
        Chain_result chain(n_steps,n);
        
        chain.sigma2[0]=sigma2;
        chain.a[0]=a;
        chain.c[0]=c;
        chain.k[0]=k;
        chain.tau2[0]=tau_0;
        
        Eigen::VectorXd beta{{ Vmu0_beta(0,0)/V0_inv_beta(0,0) }};
        chain.BETA[0]=beta;
        
        Eigen::VectorXd w;
        w.resize(n);
        for(size_t i=0; i<n; i++){
            w(i)=w0(i);
            chain.W[0](i)=w0(i);
        }
        
        double alpha_tau=alpha_tau0+n/2;
        
        auto t1 = std::chrono::high_resolution_clock::now();
       
        COV=update_COV(sigma2, a, c, k);
        auto t2 = std::chrono::high_resolution_clock::now();
       
        set_neighbors(n_neighbor);
        compute_E();
        auto t3 = std::chrono::high_resolution_clock::now();
       
        if(verbose){
        auto covv = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
        auto Ee = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2);
        std::cout<<"cov: "<<covv.count()<<std::endl;
        std::cout<<"E: "<<Ee.count()<<std::endl;
        }
        
        // --------      for n_steps:      -------
        
        for(size_t i=1; i<n_steps; i++){
            
            auto tin = std::chrono::high_resolution_clock::now();
            if(verbose){std::cout << "--------------- MCMC step number: " <<i<<" ---------------"<<std::endl;}
            if(verbose_state){std::cout<<i<<" " ;}
            
            //theta={sigma2,a,c,k}
            
            // ---------- compute N and U sets ------------
            
            auto t4 = std::chrono::high_resolution_clock::now();
            update_N();
            auto t5 = std::chrono::high_resolution_clock::now();
            update_U();
            auto t6 = std::chrono::high_resolution_clock::now();
            
            // ---------- sample beta ----------
            beta=sample_beta(V0_inv_beta, Vmu0_beta, chain.tau2[i-1], w, verbose);
            chain.BETA[i]=beta;
            //chain.BETA[i]=chain.BETA[i-1];
         
            
            // ---------- sample tau ----------
            //chain.tau2[i]=chain.tau2[i-1];
            chain.tau2[i]=sample_tau(alpha_tau, beta_tau0, w, beta,verbose);
            
            
            // ---------- sample w ----------
            auto t7 = std::chrono::high_resolution_clock::now();
            C=compute_Cnl(COV);
            auto t8 = std::chrono::high_resolution_clock::now();
            Cn=compute_Cnn(COV);
            auto t8b = std::chrono::high_resolution_clock::now();
    
            A=compute_A(COV,C,Cn);
            auto t9 = std::chrono::high_resolution_clock::now();
            
            f=compute_f(A,C,COV);
            auto t10 = std::chrono::high_resolution_clock::now();
            
            update_b(A);
            auto t11 = std::chrono::high_resolution_clock::now();
            
            auto t12 = std::chrono::high_resolution_clock::now();
            w= sample_w_r(A,f,beta,chain.tau2[i], w);
            auto t13 = std::chrono::high_resolution_clock::now();
            
        
            if(verbose){
                auto Ce = std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7);
                auto Cn = std::chrono::duration_cast<std::chrono::milliseconds>(t8b-t8);
                auto Ae = std::chrono::duration_cast<std::chrono::milliseconds>(t9-t8b);
                auto fe = std::chrono::duration_cast<std::chrono::milliseconds>(t10-t9);
                auto be = std::chrono::duration_cast<std::chrono::milliseconds>(t11-t10);
                auto We = std::chrono::duration_cast<std::chrono::milliseconds>(t13-t12);
                auto Ne = std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4);
                auto Ue = std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5);
               
                std::cout<<"Times of computation: "<<std::endl;
                std::cout<<"N: "<<Ne.count()<<std::endl;
                std::cout<<"U: "<<Ue.count()<<std::endl;
                std::cout<<"C: "<<Ce.count()<<std::endl;
                std::cout<<"Cnn: "<<Cn.count()<<std::endl;
                std::cout<<"A: "<<Ae.count()<<std::endl;
                std::cout<<"f: "<<fe.count()<<std::endl;
                std::cout<<"b: "<<be.count()<<std::endl;
                std::cout<<"W sample: "<<We.count()<<std::endl;}
                
            
            // ---------- save w full vector ----------
            chain.W[i]=w;
            
            
        // ---------- MALA steps ----------
            
            chain.sigma2[i]=chain.sigma2[i-1]; //sigma2_0
            chain.a[i]=chain.a[i-1]; //a_0
            chain.c[i]=chain.c[i-1]; //c_0
            chain.k[i]=chain.k[i-1]; //k_0
            
            bool check(false);
            size_t j(0);
            while(!check && j<n_MALA_steps){
                auto updated=Metropolis_step (chain.sigma2[i-1], chain.a[i-1], chain.c[i-1], chain.k[i-1],
                                              n_metropolis_parameters,
                                              stepsize_metropolis,
                                                                      
                                              alpha_sigma2, beta_sigma2,
                                              
                                              max_a, min_a,
                                              max_c, min_c,
                                              w,
                                              verbose);
                
                    if(updated.second){
                        COV=COV_new;
                        if(verbose){std::cout<<std::endl;std::cout << "Metropolis updated at " <<j+1<<"th trial"<< std::endl;}
                    
                        chain.sigma2[i]=updated.first[0]; //sigma2_0
                        chain.a[i]=updated.first[1]; //a_0
                        chain.c[i]=updated.first[2]; //c_0
                        chain.k[i]=updated.first[3]; //k_0
                        check=true;}
                    j++;}
            
            if(verbose){
            std::cout<<std::endl;
            std::cout<<"Current metropolis parameters:"<<std::endl;
            std::cout<<"sigma: "<<chain.sigma2[i]<<" a: "<<chain.a[i]<<" c: "<<chain.c[i]<<" k: "<<chain.k[i]<<std::endl;
            std::cout<<std::endl;
            auto tend = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend-tin);
            std::cout<<"Last step took "<<time.count()/1000<<" seconds"<<std::endl;
            std::cout<<std::endl;
            std::cout<<std::endl;}
            
        }
        
        return chain;
    }

    
    std::pair<double,Eigen::VectorXd> predict(std::array<double,3> point, Chain_result chain){
     
        Eigen::VectorXd space_distance; space_distance.resize(data.space_size+1);
        for(size_t i=0; i<data.space_size; i++)
        {
            space_distance[i]=data.s_dist(i,point);
        }
        space_distance[data.space_size]=0;
        
        
        size_t time=data.time_index(point[0]);
        size_t space=data.space_index( std::make_pair(point[1],point[2]));
    
        auto Ee=compute_E_pred(space_distance,time,space);
        
        Eigen::VectorXd result; result.resize(chain.steps);
        
        for(size_t i=0; i<chain.steps;i++){
            auto CVV=update_COV_pred(chain.sigma2[i], chain.a[i], chain.c[i], chain.k[i], point);
            auto Nn=compute_N_pred(data.space_size, CVV, Ee);
            auto W_n=w_N_l(Nn,chain.W[i]);
            auto CNN=Cnn_pred(Nn, CVV);
            auto CNL=Cnl_pred(data.space_size, Nn, CVV);
            auto A_N=Anl(CNL, CNN);
            auto fl=f_l(data.space_size, A_N, CVV, CNL);
            
            std::random_device rd;
            std::mt19937 eng(rd());
            std::normal_distribution<> dis{A_N.dot(W_n),fl};
            
            double w_sample=dis(eng);
            
            std::random_device rd1;
            std::mt19937 eng1(rd1());
            
            double mean=chain.BETA[i](0);
            mean+=w_sample;
            
            std::normal_distribution<> dis1{mean,chain.tau2[i]};
            
            result[i]= dis1(eng1);
            
        }
        double meannn(0);
        for(auto &i:result)
            meannn+=i;
        meannn/=result.size();
        
        return std::make_pair(meannn,result);
    }
    
    
    
};




#endif /* Solver_hpp */
