source("sunspots_functions.R")
library(forecast)

# multistep forecasts, as per https://robjhyndman.com/hyndsight/rolling-forecasts/
# 2 variants:
# - reestimate model as new data point comes in
# - re-select complete model as new data point comes in
# we keep the complete training set (as would be realistic in a real-world scenario)
# however, this gives ARIMA an advantage that LSTM does not get
forecast_rolling <-
  function(fit, n_forecast, train, test, fmode = "reestimate_only") {
    n <- length(test) - n_forecast + 1
    order <- arimaorder(fit)
    predictions <- matrix(0, nrow = n, ncol = n_forecast)
    lower <- matrix(0, nrow = n, ncol = n_forecast)
    upper <- matrix(0, nrow = n, ncol = n_forecast)
    
    for (i in 1:n) {
      x <- c(train, test[0:(i - 1)])
      if (fmode == "reestimate_only") {
        # re-estimate parameters, given model
        if (!is.na(order[7])) {
          refit <-
            Arima(x,
                  order = order[1:3],
                  seasonal = list(order = order[4:6], period = order[7]))
        } else {
          refit <- Arima(x, order = order[1:3],  seasonal = order[4:6])
        }
      } else if (fmode == "recompute_model") {
        # re-select the whole model
        refit <- auto.arima(x)
      }
      predictions[i,] <- forecast(refit, h = n_forecast)$mean
      lower[i,] <-
        unclass(forecast(refit, h = n_forecast)$lower)[, 2] # 95% prediction interval
      upper[i,] <-
        unclass(forecast(refit, h = n_forecast)$upper)[, 2] # 95% prediction interval
    }
    
    list(predictions = predictions,
         lower = lower,
         upper = upper)
  }


sun_spots <- datasets::sunspot.month %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)

periods_train <- 12 * 100
periods_test  <- 12 * 50
skip_span     <- 12 * 20

rolling_origin_resamples <- rolling_origin(
  sun_spots,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

split    <- rolling_origin_resamples$splits[[6]]
split_id <- rolling_origin_resamples$id[[6]]
df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(df_trn %>% add_column(key = "training"),
                df_tst %>% add_column(key = "testing")) %>%
  as_tbl_time(index = index)

rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

print(center_history)
print(scale_history)

train_vals <- df_processed_tbl %>%
  filter(key == "training") %>%
  select(value) %>%
  pull()
test_vals <- df_processed_tbl %>%
  filter(key == "testing") %>%
  select(value) %>%
  pull()

fit <- auto.arima(train_vals)
fit

n_timesteps <- 120

preds_list <-
  #  forecast_rolling(fit, n_timesteps, train_vals, test_vals, "recompute_model")
  forecast_rolling(fit, n_timesteps, train_vals, test_vals)
  
  pred_test <- drop(preds_list$predictions)
dim(pred_test)

pred_test <- (pred_test * scale_history + center_history) ^ 2

compare_test <- df_tst

for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(rep(NA, i - 1),
                                       pred_test[i,],
                                       rep(
                                         NA, nrow(compare_test) - n_timesteps - i + 1
                                       )))
}

compare_test[1:10, c(2, 4:8)] %>% print()

coln <- colnames(compare_test)[3:ncol(compare_test)]
cols <- map(coln, quo(sym(.)))
rmse_test <-
  map_dbl(cols, function(col)
    rmse(
      compare_test,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

print(rmse_test)

ggplot(compare_test, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_test1), color = "cyan") +
  geom_line(aes(y = pred_test70), color = "red") +
  geom_line(aes(y = pred_test140), color = "green") +
  geom_line(aes(y = pred_test210), color = "violet") +
  geom_line(aes(y = pred_test280), color = "red") +
  geom_line(aes(y = pred_test350), color = "green") +
  geom_line(aes(y = pred_test420), color = "violet") +
  geom_line(aes(y = pred_test481), color = "cyan")
