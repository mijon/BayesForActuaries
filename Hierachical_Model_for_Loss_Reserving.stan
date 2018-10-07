data {
  int<lower=0> N; // total number of rows
  real cum[N]; // cumulative paid
  real dev[N]; // development period
  int<lower=0> n_origin; // number of origin years
  int<lower=1, upper=n_origin> origin[N]; // origin years
  // Treat future payments as missing data, see BUGS book:
  // http://www.mrc-bsu.cam.ac.uk/wp-content/uploads/bugsbook_chapter9.pdf, page 194
  // and Stan Manual
  int<lower=0> N_mis; // number of rows of prediction data set
  real dev_mis[N_mis]; // development periods to predict
  int<lower=1, upper=n_origin> origin_mis[N_mis]; // origin periods to predict
}
parameters {
  real<lower=0> theta; // scale parameter
  real<lower=0> omega; // shape parameter 
  real<lower=0> ult[n_origin]; // ultimate loss per origin period
  real<lower=0> mu_ult; // mean ultimate loss across origin periods
  real<lower=0> sigma; // process error
  real<lower=0> sigma_ult; // random error
  real cum_mis[N_mis]; // cumulative paid to predict
}
model {
  real mu[N];
  real mu_mis[N_mis];
  real disp_sigma[N];
  real disp_sigma_mis[N_mis];
  // Priors
  theta ~ normal(46, 10); // scale parameter
  omega ~ normal(1, 2); // shape parameter 
  mu_ult ~ normal(5000, 1000);
  sigma ~ cauchy(0, 5);
  sigma_ult ~ cauchy(0, 5);
  // Hyperparameters: Modelled parameters
  ult ~ normal(mu_ult, sigma_ult);
  for (i in 1:N){ 
    mu[i] = ult[origin[i]] * weibull_cdf(dev[i], omega, theta);
    disp_sigma[i] = sigma * sqrt(mu[i]);
  }
  
  for (i in 1:N_mis){ 
    mu_mis[i] = ult[origin_mis[i]] * weibull_cdf(dev_mis[i], omega, theta);
    disp_sigma_mis[i] = sigma * sqrt(mu_mis[i]);
  }
  
  // Likelihood: Modelled data
  cum ~ normal(mu, disp_sigma); 
  cum_mis ~ normal(mu_mis, disp_sigma_mis); 
}
generated quantities{
  real Y_mean[N + N_mis];
  real Y_pred[N + N_mis];
  for(i in 1:N){
    // Posterior parameter distribution of the mean
    Y_mean[i] = ult[origin[i]] * weibull_cdf(dev[i], omega, theta);
    // Posterior predictive distribution
    Y_pred[i] = normal_rng(Y_mean[i], sigma*sqrt(Y_mean[i]));   
  }
  for(i in 1:N_mis){
    // Posterior parameter distribution of the mean
    Y_mean[N + i] = ult[origin_mis[i]] * weibull_cdf(dev_mis[i], omega, theta);
    // Posterior predictive distribution
    Y_pred[N + i] = normal_rng(Y_mean[N + i], sigma*sqrt(Y_mean[N + i]));   
  }
}