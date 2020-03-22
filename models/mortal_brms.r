library(feather)
library(tidyverse)
library(data.table)
library(brms)
library(bayesplot)
library(tidybayes)

setwd("/Users/wbeard/repos/crash/notebooks/")
pl <- function(ct, fn) {
  fn <- paste0("/Users/wbeard/repos/crash/reports/figures/", fn)
  pdf(file = fn,
      width = 24, # The width of the plot in inches
      # height = 4
  )
  plot(ct)
  dev.off()
}

pth <- 'covid/data/mort_0320.fth'
pth_sim <- 'covid/data/mort_0320_sim.fth'
df = read_feather(pth) %>% as.data.frame()

head(df)

# Model
f = (bf(ldeaths ~ daysi + (1 | state))) + gaussian()
mdeath1 <- brm(f, df, cores = 4, control = list(adapt_delta = .995))

f = bf(ldeaths ~ dayrate + (1 | state), dayrate ~ state) + poisson()
make_stancode(f, df)
mdeath2 <- brm(f, df, cores = 4, control = list(adapt_delta = .995))

# , max_treedepth = 13


# marginal_effects
post <- marginal_effects(mdeath1, method = 'predict')
p2 <- plot(post, points = TRUE, plot = FALSE)

# pp_check
pp_check(mdeath1, nsamples = 50)

# Predictions
get_preds <- function(mod, prefix, df_new_base) {
  df <- df_new_base %>%
    add_predicted_draws(mod, prediction = "pred") %>%
    ungroup() %>%
    select(row, pred, state, daysi, .row) %>%
    group_by(row) %>%
    summarise(mu=mean(pred), q10=quantile(pred, p=.1), q90=quantile(pred, p=.9))
  new_names <- paste(prefix, colnames(df), sep="_")
  new_names[1] <- "row"
  colnames(df) <- new_names
  df
}

dfsim = read_feather(pth_sim) %>% as.data.frame()

dfsim_out = get_preds(mdeath1, 'pred', dfsim)
write_feather(dfsim_out, 'covid/data/mort_0320_sim_out.fth')

# dfsim$row <- 1:nrow(dfsim)
