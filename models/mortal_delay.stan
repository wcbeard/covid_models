/*
Estimate infection rate as auto-regressive
  time series.
N: # rows
perc_delayed: infection rate (positive / total) ~4 days ago
perc_delayed_prev: infection rate the day before that
p_death_baseline: lowest state's prob(death)
*/

data {
    int N;
    // int S;
    int<lower=1> npos_delayed[N];
    int<lower=1> ntot_delayed[N];
    int<lower=1> n_deaths[N];
    // int<lower=1, upper=S> state[N];
    real<lower=0, upper=1> perc_delayed[N];
    real<lower=0, upper=1> perc_delayed_prev[N];
    real<lower=0, upper=1> p_death_baseline[N];
}

parameters {
    real<lower=0> p_sig;
    vector<lower=0, upper=1>[N] p_delay;
    // vector<lower=0, upper=1>[N] p_death;
    // vector[N] p_death_logit;
    vector[N] b_death;
    // vector<lower=0, upper=1>[N] prob_death;
}

model {
    
    vector[N] prob_death;
    p_delay ~ normal(perc_delayed_prev, p_sig);
    p_sig ~ normal(0, 2);
    // p_inf ~ normal(p_state_prev, p_sig);
    npos_delayed ~ binomial(ntot_delayed, p_delay);

    
    prob_death = inv_logit(b_death .* p_delay);
    // inv_logit(alpha + x[n] * beta);
    n_deaths ~ binomial(npos_delayed, prob_death);
}