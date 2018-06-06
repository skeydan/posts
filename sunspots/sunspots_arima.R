source("sunspots_functions.R")
library(forecast)

# multistep forecasts, as per https://robjhyndman.com/hyndsight/rolling-forecasts/ 
# 2 variants:
# - reestimate model as new data point comes in
# - re-select complete model as new data point comes in 
# we keep the complete training set (as would be realistic)
forecast_rolling <- function(fit, n_forecast, train, test, fmode = "reestimate_only") {
  
  n <- length(test) - n_forecast + 1
  order <- arimaorder(fit)
  predictions <- matrix(0, nrow=n, ncol= n_forecast)
  lower <- matrix(0, nrow=n, ncol= n_forecast) 
  upper <- matrix(0, nrow=n, ncol= n_forecast)
  
  for(i in 1:n) {  
    x <- c(train, test[0:(i-1)])
    if(fmode == "reestimate_only") {  # re-estimate parameters, given model 
      # important: must also pass in the period because this information gets lost when converting ts to vectors in concatenation step above
      if(!is.na(order[7])) {
        refit <- Arima(x, order=order[1:3],  seasonal=list(order = order[4:6], period = order[7]))
      } else {
        refit <- Arima(x, order=order[1:3],  seasonal = order[4:6])
      }
    } else if (fmode == "recompute_model") { # re-select the whole model
      refit <- auto.arima(x)
    }
    predictions[i,] <- forecast(refit, h = n_forecast)$mean
    lower[i,] <- unclass(forecast(refit, h = n_forecast)$lower)[,2] # 95% prediction interval
    upper[i,] <- unclass(forecast(refit, h = n_forecast)$upper)[,2] # 95% prediction interval
  }
  
  list(predictions = predictions, lower = lower, upper = upper)
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

fit <- auto.arima(df_trn$value)
fit

n_timesteps <- 120

preds_list <- forecast_rolling(fit, n_timesteps, df_trn$value, df_tst$value)
pred_test <- drop(preds_list$predictions)
dim(pred_test)

compare_test <- df_tst

for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(
      rep(NA, i - 1),
      pred_test[i,],
      rep(NA, nrow(compare_test) - n_timesteps - i + 1)
    ))
}

compare_test[1:10, c(2, 4:8)] %>% print()

coln <- colnames(compare_test)[3:ncol(compare_test)]
cols <- map(coln, quo(sym(.)))
rsme_test <-
  map_dbl(cols, function(col)
    rmse(
      compare_test,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

print(rsme_test)

ggplot(compare_test, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_test1), color = "cyan") +
  geom_line(aes(y = pred_test100), color = "red") +
  geom_line(aes(y = pred_test200), color = "green") +
  geom_line(aes(y = pred_test300), color = "violet")

