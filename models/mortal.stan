data {
    int K;               // number of branches
    int<lower=1> nret[K];
    int<lower=1> ntot[K];
}

parameters {
    real<lower=0> theta_k[K];
}

model {
    theta_k ~ beta(4, 4);
    nret ~ binomial(ntot, theta_k);
}