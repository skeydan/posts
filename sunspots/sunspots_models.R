
source("sunspots_functions.R")


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
df_val <- training(rolling_origin_resamples$splits[[5]])
df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing"),
  df_val %>% add_column(key = "validation")
) %>%
  as_tbl_time(index = index)

rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

model_exists <- FALSE
stateful <- FALSE
batch_size   <- 1
n_timesteps <- 12
n_predictions <- n_timesteps
n_features <- 1
n_epochs  <- 300
n_units <- 64
dropout <- 0.2
recurrent_dropout <- 0.2

model_path <- file.path(
  "models",
  paste0(
    "LSTM_stateful_",
    stateful,
    "_tsteps_",
    n_timesteps,
    "_epochs_",
    n_epochs,
    "_units_",
    n_units,
    "_batchsize_",
    batch_size,
    "_dropout_",
    dropout,
    "_recdrop_",
    recurrent_dropout,
    ".hdf5"
  )
)

train_vals <- df_processed_tbl %>%
  filter(key == "training") %>%
  select(value) %>%
  pull()
test_vals <- df_processed_tbl %>%
  filter(key == "testing") %>%
  select(value) %>%
  pull()
valid_vals <- df_processed_tbl %>%
  filter(key == "validation") %>%
  select(value) %>%
  pull()

train_matrix <-
  build_matrix(train_vals, n_timesteps + n_predictions)
test_matrix <- build_matrix(test_vals, n_timesteps + n_predictions)
valid_matrix <-
  build_matrix(valid_vals, n_timesteps + n_predictions)

X_train <- train_matrix[, 1:n_timesteps]
y_train <- train_matrix[, (n_timesteps + 1):(n_timesteps * 2)]

X_test <- test_matrix[, 1:n_timesteps]
y_test <- test_matrix[, (n_timesteps + 1):(n_timesteps * 2)]

X_valid <- valid_matrix[, 1:n_timesteps]
y_valid <- valid_matrix[, (n_timesteps + 1):(n_timesteps * 2)]

X_train <- reshape_X_3d(X_train)
X_test <- reshape_X_3d(X_test)
X_valid <- reshape_X_3d(X_valid)

y_train <- reshape_X_3d(y_train)
y_test <- reshape_X_3d(y_test)
y_valid <- reshape_X_3d(y_valid)

if (!model_exists) {
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units            = n_units,
      batch_input_shape  = c(batch_size, n_timesteps, n_features),
      return_sequences = TRUE
    ) %>%
    time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(loss = 'mean_squared_error', optimizer = 'adam')
  
  if (!stateful) {
    model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_valid, y_valid),
      batch_size = batch_size,
      epochs     = n_epochs,
      callbacks = list(callback_early_stopping(patience = 10))
    )
    
  } else {
    for (i in 1:n_epochs) {
      model %>% fit(
        x          = X_train,
        y          = y_train,
        validation_data = list(X_valid, y_valid),
        callbacks = list(callback_early_stopping(patience = 2)),
        batch_size = batch_size,
        epochs     = 1,
        shuffle    = FALSE
      )
      
      model %>% reset_states()
    }
  }
  model %>% save_model_hdf5(filepath = model_path)
  
  
} else {
  model <- load_model_hdf5(model_path)
}

model

if (stateful)
  model %>% reset_states()

# Make Predictions
pred_out <- model %>%
  predict(X_test, batch_size = batch_size) %>%
  .[, , 1]

# Retransform values
pred_out <- (pred_out * scale_history + center_history) ^ 2
pred_out[1:10, 1:5] %>% print()

compare_df <- df %>% filter(key == "testing")

for (i in 1:nrow(pred_out)) {
  varname <- paste0("pred_test", i)
  compare_df <-
    mutate(compare_df,!!varname := c(
      rep(NA, n_timesteps + i - 1),
      pred_out[i,],
      rep(NA, nrow(compare_df) - n_timesteps * 2 - i + 1)
    ))
}

compare_df %>% write_csv(str_replace(model_path, "hdf5", "csv") )

compare_df[n_timesteps:(n_timesteps + 10), c(2, 4:8)] %>% print()

# multiple_rmse <-
#   calc_multiple_rmse(compare_df %>% select(-c(index, key)))

coln <- colnames(compare_df)[4:ncol(compare_df)]
cols <- map(coln, quo(sym(.)))
multiple_rmse <-
  map_dbl(cols, function(col)
    rmse(
      compare_df,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

print(multiple_rmse)

ggplot(compare_df, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_test1), color = "cyan") + 
  geom_line(aes(y = pred_test100), color = "red") + 
  geom_line(aes(y = pred_test200), color = "green") + 
  geom_line(aes(y = pred_test300), color = "violet") +
  geom_line(aes(y = pred_test400), color = "red") + 
  geom_line(aes(y = pred_test500), color = "green") %>%  
  print()
