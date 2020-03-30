library(feather)
library(tidyverse)
library(data.table)
library(brms)
library(bayesplot)
library(tidybayes)


setwd("/Users/wbeard/repos/covid/data/")
pl <- function(ct, fn) {
  fn <- paste0("/Users/wbeard/repos/crash/reports/figures/", fn)
  pdf(file = fn,
      width = 24, # The width of the plot in inches
      # height = 4
  )
  plot(ct)
  dev.off()
}

day <- '0329'
fn_state = paste0('mort_m2_coef', day, '.fth')
fn_preds = paste0('mort_preds_', day, '.fth')
fn_spread <- 'mort_draws.fth'

fn_data = paste0('mort_', day, '.fth')
# pth <- 'mort_0327.fth'
fn_sim <- paste0('mort_sim_', day, '.fth')
df = read_feather(fn_data) %>% as.data.frame()
nrow(df)
# head(df)
# make_stancode(f, df)

# Model
# f = bf(ldeaths ~ daysi + (1 | state)) + gaussian()
# mdeath1 <- brm(f, df, cores = 4, control = list(adapt_delta = .995))

f = bf(ldeaths ~ (1 | state) + daysi + (0 + daysi | state))
mdeath3 <- brm(f, df, prior = c(
  prior(normal(0, 2), class = b)
  # prior(normal(0, 2), class = sigma),
  # prior(normal(0, 2), class = Intercept)
), cores = 4, control = list(adapt_delta = .995, max_treedepth = 12))
mdeath3
state <- coef(mdeath3)$state[,,2]
state <- data.frame(state = row.names(state), state)
write_feather(as.data.frame(state), fn_state)

spread <- mdeath3 %>% spread_draws(b_daysi, r_state[state, coef])
spread <- spread[spread$coef == 'daysi',]
write_feather(spread, fn_spread)

# Predictions
get_preds <- function(mod, prefix, df_new_base) {
  df <- df_new_base %>%
    add_predicted_draws(mod, prediction = "pred") %>%
    ungroup() %>%
    select(row, pred, state, daysi, .row) %>%
    group_by(state, daysi) %>%
    summarise(mu=mean(pred), q10=quantile(pred, p=.1), q90=quantile(pred, p=.9))
  new_names <- paste(prefix, colnames(df), sep="_")
  # new_names[1] <- "row"
  colnames(df) <- new_names
  df
}

dfsim = read_feather(fn_sim) %>% as.data.frame()
dfsim_out = get_preds(mdeath3, 'pred', dfsim)
head(dfsim_out)
write_feather(dfsim_out, fn_preds)
# write_feather(dfsim_out, 'mort_v2_sim_out.fth')

# dfsim$row <- 1:nrow(dfsim)



m <- brm(bf(ldeaths ~ (1 + daysi | state)), prior =  data = df)
get_prior(f, data = df)

get_prior(bf(ldeaths ~ (1 | state) + (0 + daysi | state)), data = df)
get_prior(bf(ldeaths ~ (1 | state) + daysi + (0 + daysi | state)), data = df)
# , max_treedepth = 13


# marginal_effects
post <- marginal_effects(mdeath1, method = 'predict')
p2 <- plot(post, points = TRUE, plot = FALSE)

# pp_check
pp_check(mdeath1, nsamples = 50)
