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
df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
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

print(center_history)
print(scale_history)

model_exists <- FALSE


FLAGS <- flags(
  flag_boolean("stateful", FALSE),
  flag_boolean("stack_layers", FALSE),
  flag_integer("batch_size", 10),
  flag_integer("n_timesteps", 12),
  flag_integer("n_epochs", 100),
  flag_numeric("dropout", 0.2),
  flag_numeric("recurrent_dropout", 0.2),
  flag_string("loss", "logcosh"),
  flag_string("optimizer_type", "sgd"),
  flag_integer("n_units", 128),
  flag_numeric("lr", 0.003),
  flag_numeric("momentum", 0.9),
  flag_integer("patience", 20)
)

n_predictions <- FLAGS$n_timesteps
n_features <- 1

optimizer <- switch(FLAGS$optimizer_type,
                    sgd = optimizer_sgd(lr = FLAGS$lr, momentum = FLAGS$momentum))
callbacks <- list(
   #callback_learning_rate_scheduler(function(epoch, lr) lr + epoch * 0.001)
  callback_early_stopping(patience = FLAGS$patience)
#  callback_tensorboard(log_dir = "/tmp/tf", 
#                       histogram_freq = 5,
#                       batch_size = batch_size,
#                       write_grads = TRUE,
#                       write_images = TRUE
)

model_path <- file.path(
  "models",
  paste0(
    "LSTM_stateful_",
    FLAGS$stateful,
    "_tsteps_",
    FLAGS$n_timesteps,
    "_epochs_",
    FLAGS$n_epochs,
    "_units_",
    FLAGS$n_units,
    "_batchsize_",
    FLAGS$batch_size,
    "_dropout_",
    FLAGS$dropout,
    "_recdrop_",
    FLAGS$recurrent_dropout,
    "_stack_",
    FLAGS$stack_layers,
    "_loss_",
    FLAGS$loss,
    "_optimizer_",
    FLAGS$optimizer_type,
    "_lr_",
    FLAGS$lr,
    "_momentum_",
    FLAGS$momentum,
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

train_matrix <-
  build_matrix(train_vals, FLAGS$n_timesteps + n_predictions)
test_matrix <- build_matrix(test_vals, FLAGS$n_timesteps + n_predictions)

X_train <- train_matrix[, 1:FLAGS$n_timesteps]
y_train <- train_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
X_train <- X_train[1:(nrow(X_train) %/% FLAGS$batch_size * FLAGS$batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% FLAGS$batch_size * FLAGS$batch_size), ]

X_test <- test_matrix[, 1:FLAGS$n_timesteps]
y_test <- test_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
X_test <- X_test[1:(nrow(X_test) %/% FLAGS$batch_size * FLAGS$batch_size), ]
y_test <- y_test[1:(nrow(y_test) %/% FLAGS$batch_size * FLAGS$batch_size), ]

X_train <- reshape_X_3d(X_train)
X_test <- reshape_X_3d(X_test)

y_train <- reshape_X_3d(y_train)
y_test <- reshape_X_3d(y_test)


if (!model_exists) {
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units            = FLAGS$n_units,
      batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
      dropout = FLAGS$dropout,
      recurrent_dropout = FLAGS$recurrent_dropout,
      return_sequences = TRUE
    )
  if (FLAGS$stack_layers) {
    
    model %>%
      layer_lstm(
        units            = FLAGS$n_units,
        dropout = FLAGS$dropout,
        recurrent_dropout = FLAGS$recurrent_dropout,
        return_sequences = TRUE
      )
  }
  model %>% time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(loss = FLAGS$loss, optimizer = optimizer, metrics = list("mean_squared_error"))
  
  print(model)
  
  if (!FLAGS$stateful) {
    history <- model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_test, y_test),
      batch_size = FLAGS$batch_size,
      epochs     = FLAGS$n_epochs,
      callbacks = callbacks
    )
    
  } else {
    for (i in 1:n_epochs) {
      history <- model %>% fit(
        x          = X_train,
        y          = y_train,
        validation_data = list(X_test, y_test),
        callbacks = callbacks,
        batch_size = FLAGS$batch_size,
        epochs     = 1,
        shuffle    = FALSE)
      
      
      model %>% reset_states()
    }
  }
  model %>% save_model_hdf5(filepath = model_path)
  
  
} else {
  model <- load_model_hdf5(model_path)
}

model

if (FLAGS$stateful)
  model %>% reset_states()

plot(history, metrics = "loss")

# Train predictions -------------------------------------------------------

pred_train <- model %>%
  predict(X_train, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Retransform values
pred_train <- (pred_train * scale_history + center_history) ^2
pred_train[1:10, 1:5] %>% print()
compare_train <- df %>% filter(key == "training")

for (i in 1:nrow(pred_train)) {
  varname <- paste0("pred_train", i)
  compare_train <-
    mutate(compare_train,!!varname := c(
      rep(NA, FLAGS$n_timesteps + i - 1),
      pred_train[i,],
      rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
    ))
}

compare_train %>% write_csv(str_replace(model_path, ".hdf5", ".train.csv"))
compare_train[FLAGS$n_timesteps:(FLAGS$n_timesteps + 10), c(2, 4:8)] %>% print()

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
  geom_line(aes(y = pred_train200), color = "green") +
  geom_line(aes(y = pred_train300), color = "violet") +
  geom_line(aes(y = pred_train400), color = "cyan") +
  geom_line(aes(y = pred_train500), color = "red") +
  geom_line(aes(y = pred_train600), color = "red") +
  geom_line(aes(y = pred_train700), color = "green") +
  geom_line(aes(y = pred_train800), color = "violet") +
  geom_line(aes(y = pred_train900), color = "cyan") +
  geom_line(aes(y = pred_train1000), color = "red") +
  geom_line(aes(y = pred_train1100), color = "green") 



# Test predictions--------------------------------------------------------------------

pred_test <- model %>%
  predict(X_test, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Retransform values
pred_test <- (pred_test * scale_history + center_history) ^2
pred_test[1:10, 1:5] %>% print()
compare_test <- df %>% filter(key == "testing")

for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(
      rep(NA, FLAGS$n_timesteps + i - 1),
      pred_test[i,],
      rep(NA, nrow(compare_test) - FLAGS$n_timesteps * 2 - i + 1)
    ))
}

compare_test %>% write_csv(str_replace(model_path, ".hdf5", ".test.csv"))
compare_test[FLAGS$n_timesteps:(FLAGS$n_timesteps + 10), c(2, 4:8)] %>% print()

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
  geom_line(aes(y = pred_test50), color = "red") +
  geom_line(aes(y = pred_test100), color = "green") +
  geom_line(aes(y = pred_test150), color = "violet") +
  geom_line(aes(y = pred_test200), color = "cyan") +
  geom_line(aes(y = pred_test250), color = "red") +
  geom_line(aes(y = pred_test300), color = "green") +
  geom_line(aes(y = pred_test350), color = "cyan") +
  geom_line(aes(y = pred_test400), color = "red") +
  geom_line(aes(y = pred_test450), color = "green") +  
  geom_line(aes(y = pred_test500), color = "cyan") +
  geom_line(aes(y = pred_test550), color = "violet")

