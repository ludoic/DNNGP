//
//  main.cpp
//  DNNGP_CLASSES
//
//  Created by Ludovico Italo Casati and Alexandre Pugin
//


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "Data_structure.hpp"

#include "Solution.hpp"

#include "Solver.hpp"

//
//
//
//
//  Example of usage of the DNNGP classes
//
//
//
//


int main(int argc, const char * argv[]) {
    
    
    // Define all the needed parameters
    
    constexpr size_t time_size=10;
    constexpr size_t space_size=49;
    constexpr size_t n=time_size*space_size;
    
    // The data is necessarily CSV with no header
    // The format is: coord1,coord2,time,response
    
    std::string dataset_file="some_data.txt";
     
    size_t n_covariates=0;
    
    
    // ----------------------   construct SPdata and DNNGP object  -------------------
    
    // SPdata<Type,Size>()   !!! In the current version Type must be = 3 !!!
    
    SPdata<3,n> data(space_size, time_size, n_covariates, dataset_file);
    
    // construct a DNNGP object named "model", calling the data
    
    DNNGP<3,n> model(data);
    
    
    
    // ----------------------   Set parameters  -------------------------------------
    
    // Define the design matrix
    Eigen::MatrixXd X = Eigen::MatrixXd::Constant(n, 1, 1.0);
    
    // Or get it from data
    auto data_covariates=data.get_covariates();
    
    // call .set_X to set the design matrix
    model.set_X(X);
    
    
    // if necessary extract data mean
    double emp_mean=data.mean();
    
    
    
    
    // ---  BETA parameters  ---
    
    Eigen::MatrixXd V_inv; // precision of beta
    V_inv.resize(1,1); V_inv(0,0)=1/(100*data.var()/std::sqrt(n));
    
    // Vmu0= V_inv * mu
    Eigen::VectorXd Vmu0 = Eigen::VectorXd::Ones(1);
    Vmu0 = emp_mean*V_inv(0,0) * Vmu0 ;

    
    
    
    // --- Initialize w  ---
    
    model.initialize_w();         // set w_i=0
    model.initialize_w(emp_mean); // set w_i=y_i-emp_mean
    
    
    // Other parameters:
    size_t n_neighbor=16;

    //    MCMC paramters
    size_t n_chain_steps(2000);
    size_t n_MALA_steps(6);
    
    //    Tau^2 prior parameters
    double alpha_tau0(11);
    double beta_tau0(1);
    double tau_0(std::sqrt(0.1));   // NB: tau_0 = sqrt(tau^2), First value of the chain
    
    
    //    Sigma^2
    double alpha_sigma2(101);
    double beta_sigma2(100);
    double sigma2(1);               // First value of the chain
    
    // a, c and k
    double max_a(55);
    double min_a(45);
    double max_c(30);
    double min_c(20);
    
    double a(50);                   // First value of the chain
    double c(25);                   // First value of the chain
    double k(0.75);                 // First value of the chain

    //metropolis step
    double stepsize_metropolis(0.15);
    size_t dimention_parameters_space(4);

    // verbose
    bool hyper_verbose(true); // really verbose
    bool step_verbose(false); //print current step number only
    
    
    
    // -------------------------------------  Run the chain  -------------------------------------
    
    
    // call .run_MCMC with the parameters, the output is an object of type Chain_result
    Chain_result results = model.run_MCMC(n_chain_steps, n_MALA_steps, n_neighbor,
                                          sigma2, a, c, k,
                                          alpha_tau0, beta_tau0,
                                          Vmu0, V_inv,
                                          alpha_sigma2, beta_sigma2,
                                          max_a, min_a,
                                          max_c, min_c,
                                          tau_0,
                                          dimention_parameters_space, stepsize_metropolis,hyper_verbose,step_verbose);

    
    
    // to save just run .save_to_file(filename) on the Chain_result object
    results.save_to_file("hopefully_a_good_traceplot.txt");
    
    
    
    
    // --------------------       predict      --------------------
    
    // point should be (time, space1, space2)
    std::array<double,3> new_point{11.0, 12334.0, 12453.0};
   
    // input is of type:   std::array<double,3>, Chain_result
    
    auto y_new=model.predict(new_point, results);
    
    // output is pair (prediction_mean,prediction_chain)

    // the prediction is not currently computationally optimized
    

    
    return 0;
}
