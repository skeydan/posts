


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
stack_layers <- TRUE
batch_size   <- 1
n_timesteps <- 120
n_predictions <- n_timesteps
n_features <- 1
n_epochs  <- 300
n_units <- 64
dropout <- 0.4
recurrent_dropout <- 0.4
loss <- "logcosh"

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
    "_stack_",
    stack,
    "_loss",
    loss,
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

X_train <- X_train[1:nrow(X_train) - 1, , , drop = FALSE]
X_valid <- X_valid[1:nrow(X_valid) - 1, , , drop = FALSE]
X_test <- X_test[1:nrow(X_test) - 1, ,  , drop = FALSE]

y_train <- y_train[1:nrow(y_train) - 1, ,  , drop = FALSE]
y_valid <- y_valid[1:nrow(y_valid) - 1, ,  , drop = FALSE]
y_test <- y_test[1:nrow(y_test) - 1, ,  , drop = FALSE]


if (!model_exists) {
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units            = n_units,
      batch_input_shape  = c(batch_size, n_timesteps, n_features),
      dropout = dropout,
      recurrent_dropout = recurrent_dropout,
      return_sequences = TRUE
    )
  if (stack_layers) {
    model %>%
      layer_lstm(
        units            = n_units,
        dropout = dropout,
        recurrent_dropout = recurrent_dropout,
        return_sequences = TRUE
      )
  }
  model %>% time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(loss = loss, optimizer = 'adam', metrics = list("mean_squared_error"))
  
  if (!stateful) {
    model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_valid, y_valid),
      batch_size = batch_size,
      epochs     = n_epochs,
      callbacks = list(
        callback_early_stopping(patience = 10),
        callback_reduce_lr_on_plateau()
      )
    )
    
  } else {
    for (i in 1:n_epochs) {
      model %>% fit(
        x          = X_train,
        y          = y_train,
        validation_data = list(X_valid, y_valid),
        callbacks = list(
          callback_early_stopping(patience = 10),
          callback_reduce_lr_on_plateau()
        ),
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


# Train predictions -------------------------------------------------------

pred_train <- model %>%
  predict(X_train, batch_size = batch_size) %>%
  .[, , 1]

# Retransform values
pred_train <- (pred_train * scale_history + center_history) ^ 2
pred_train[1:10, 1:5] %>% print()
compare_train <- df %>% filter(key == "training")

for (i in 1:nrow(pred_train)) {
  varname <- paste0("pred_train", i)
  compare_train <-
    mutate(compare_train,!!varname := c(
      rep(NA, n_timesteps + i - 1),
      pred_train[i,],
      rep(NA, nrow(compare_train) - n_timesteps * 2 - i + 1)
    ))
}

compare_train %>% write_csv(str_replace(model_path, ".hdf5", ".train.csv"))
compare_train[n_timesteps:(n_timesteps + 10), c(2, 4:8)] %>% print()

coln <- colnames(compare_train)[4:ncol(compare_train)]
cols <- map(coln, quo(sym(.)))
rsme_train <-
  map_dbl(cols, function(col)
    rmse(
      compare_train,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

print(rsme_train)

ggplot(compare_train, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_train1), color = "cyan") +
  geom_line(aes(y = pred_train100), color = "red") +
  geom_line(aes(y = pred_train300), color = "green") +
  geom_line(aes(y = pred_train500), color = "violet") +
  geom_line(aes(y = pred_train700), color = "cyan") +
  geom_line(aes(y = pred_train900), color = "red")


# Test predictions--------------------------------------------------------------------

pred_test <- model %>%
  predict(X_test, batch_size = batch_size) %>%
  .[, , 1]

# Retransform values
pred_test <- (pred_test * scale_history + center_history) ^ 2
pred_test[1:10, 1:5] %>% print()
compare_test <- df %>% filter(key == "testing")

for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(
      rep(NA, n_timesteps + i - 1),
      pred_test[i,],
      rep(NA, nrow(compare_test) - n_timesteps * 2 - i + 1)
    ))
}

compare_test %>% write_csv(str_replace(model_path, ".hdf5", ".test.csv"))
compare_test[n_timesteps:(n_timesteps + 10), c(2, 4:8)] %>% print()

coln <- colnames(compare_test)[4:ncol(compare_test)]
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
