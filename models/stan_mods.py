class Mv1:
    stan_code = """
    /*
    N: # rows
    S: # states
    state: categorical
    p_inf: infection rate
    */
    data {
        int N;
        int S;
        int<lower=1> npos_delayed[N];
        int<lower=1> ntot_delayed[N];
        int<lower=1, upper> state[N];
    }

    parameters {
        real<lower=0, upper=1> p_inf;
    }

    model {
        p_inf ~ beta(1, 1);
        npos_delayed ~ binomial(ntot_delayed, p_inf);
    }
    """

    def mk_data(df):
        data = dict(
            N=len(df),
            S=df.state.nunique(),
            npos_delayed=df.pos_delayed,
            ntot_delayed=df.tot_delayed,
        )
        return data


class Mpv1:
    stan_code = """
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
    """

    def preprocess_data(ddfs):
        _ints = ["death", "total", "positive", "pos_delayed", "tot_delayed"]
        _fs = "perc_delayed prev_perc_delayed".split()
        ddelayed = (
            ddfs[["state", "date"] + _ints + _fs]
            .query("pos_delayed > 0")
            .query("prev_perc_delayed > 0")
            .assign(
                **{
                    icol: lambda x, icol=icol: x[icol].astype(int)
                    for icol in _ints
                }
            )
            .reset_index(drop=1)
        )
        return ddelayed

    def mk_data(df):
        data = dict(
            N=len(df),
            # S=df.state.nunique(),
            npos_delayed=df.pos_delayed,
            ntot_delayed=df.tot_delayed,
            perc_delayed=df.perc_delayed,
            prev_perc_delayed=df.prev_perc_delayed,
        )
        return data
