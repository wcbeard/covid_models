/*
Estimate infection rate as auto-regressive
  time series.
N: # rows
perc_delayed: infection rate (positive / total) ~4 days ago
prev_perc_delayed: infection rate the day before that
*/
data {
    int N;
    // int S;
    int<lower=1> npos_delayed[N];
    int<lower=1> ntot_delayed[N];
    // int<lower=1, upper=S> state[N];
    real<lower=0, upper=1> perc_delayed[N];
    real<lower=0, upper=1> prev_perc_delayed[N];
}

parameters {
    real<lower=0> p_sig;
    vector<lower=0, upper=1>[N] p_delay;
}

model {
    p_delay ~ normal(prev_perc_delayed, p_sig);
    p_sig ~ normal(0, 2);
    // p_inf ~ normal(p_state_prev, p_sig);
    npos_delayed ~ binomial(ntot_delayed, p_delay);
}