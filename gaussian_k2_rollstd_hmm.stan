data {
  int<lower=1> T;
  int<lower=1> d;
  int<lower=1> K;
  matrix[T, d] Y;
  vector<lower=0>[d] mu_prior_scale;
  real<lower=0> trans_conc;
  real<lower=0> log_sigma_sd;
  real<lower=0> z_sigma_sd;
}

parameters {
  array[K] simplex[K] A;
  ordered[K] log_sigma_level;
  matrix[K, d] mu;
  matrix[K, d] z_sigma;
}

transformed parameters {
  matrix[K, K] A_mat;
  matrix[K, K] logA;
  vector[K] logpi;
  matrix<lower=0>[K, d] sigma;

  for (k in 1:K) {
    for (j in 1:K) {
      A_mat[k, j] = A[k, j];
      logA[k, j] = log(A[k, j]);
    }
  }

  {
    row_vector[K] pi_row = rep_row_vector(1.0 / K, K);
    for (n in 1:200) {
      pi_row = pi_row * A_mat;
      pi_row = pi_row / sum(pi_row);
    }
    for (k in 1:K) {
      logpi[k] = log(pi_row[k]);
    }
  }

  for (k in 1:K) {
    for (j in 1:d) {
      sigma[k, j] = exp(log_sigma_level[k] + z_sigma_sd * z_sigma[k, j]);
    }
  }
}

model {
  to_vector(z_sigma) ~ std_normal();
  log_sigma_level ~ normal(0, log_sigma_sd);

  for (k in 1:K) {
    A[k] ~ dirichlet(rep_vector(trans_conc, K));
    for (j in 1:d) {
      mu[k, j] ~ normal(0, mu_prior_scale[j]);
    }
  }

  {
    matrix[T, K] log_emis;
    matrix[T, K] log_alpha;

    for (t in 1:T) {
      for (k in 1:K) {
        real lp = 0;
        for (j in 1:d) {
          lp += normal_lpdf(Y[t, j] | mu[k, j], sigma[k, j]);
        }
        log_emis[t, k] = lp;
      }
    }

    for (k in 1:K) {
      log_alpha[1, k] = logpi[k] + log_emis[1, k];
    }

    for (t in 2:T) {
      for (k in 1:K) {
        vector[K] acc;
        for (j in 1:K) {
          acc[j] = log_alpha[t - 1, j] + logA[j, k];
        }
        log_alpha[t, k] = log_emis[t, k] + log_sum_exp(acc);
      }
    }

    target += log_sum_exp(to_vector(log_alpha[T, ]'));
  }
}

generated quantities {
  vector[K] pi;
  matrix[T, K] gamma;

  {
    matrix[T, K] log_emis;
    matrix[T, K] log_alpha;
    matrix[T, K] log_beta;

    for (k in 1:K) {
      pi[k] = exp(logpi[k]);
    }

    for (t in 1:T) {
      for (k in 1:K) {
        real lp = 0;
        for (j in 1:d) {
          lp += normal_lpdf(Y[t, j] | mu[k, j], sigma[k, j]);
        }
        log_emis[t, k] = lp;
      }
    }

    for (k in 1:K) {
      log_alpha[1, k] = logpi[k] + log_emis[1, k];
    }

    for (t in 2:T) {
      for (k in 1:K) {
        vector[K] acc;
        for (j in 1:K) {
          acc[j] = log_alpha[t - 1, j] + logA[j, k];
        }
        log_alpha[t, k] = log_emis[t, k] + log_sum_exp(acc);
      }
    }

    for (k in 1:K) {
      log_beta[T, k] = 0;
    }

    for (t_rev in 1:(T - 1)) {
      int t = T - t_rev;
      for (k in 1:K) {
        vector[K] acc;
        for (j in 1:K) {
          acc[j] = logA[k, j] + log_emis[t + 1, j] + log_beta[t + 1, j];
        }
        log_beta[t, k] = log_sum_exp(acc);
      }
    }

    for (t in 1:T) {
      vector[K] lg;
      real norm;
      for (k in 1:K) {
        lg[k] = log_alpha[t, k] + log_beta[t, k];
      }
      norm = log_sum_exp(lg);
      for (k in 1:K) {
        gamma[t, k] = exp(lg[k] - norm);
      }
    }
  }
}
