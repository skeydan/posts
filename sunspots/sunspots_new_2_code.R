# Core Tidyverse
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
library(keras)
library(tfruns)


# exploration -------------------------------------------------------------


sun_spots <- datasets::sunspot.month %>%
    tk_tbl() %>%
    mutate(index = as_date(index)) %>%
    as_tbl_time(index = index)

sun_spots

p1 <- sun_spots %>%
    ggplot(aes(index, value)) +
    geom_point(color = palette_light()[[1]], alpha = 0.5) +
    theme_tq() +
    labs(
        title = "From 1749 to 2013 (Full Data Set)"
    )

p2 <- sun_spots %>%
    filter_time("start" ~ "1800") %>%
    ggplot(aes(index, value)) +
    geom_line(color = palette_light()[[1]], alpha = 0.5) +
    geom_point(color = palette_light()[[1]]) +
    geom_smooth(method = "loess", span = 0.2, se = FALSE) +
    theme_tq() +
    labs(
        title = "1749 to 1759 (Zoomed In To Show Changes over the Year)",
        caption = "datasets::sunspot.month"
    )

p_title <- ggdraw() + 
    draw_label("Sunspots", size = 18, fontface = "bold", colour = palette_light()[[1]])

plot_grid(p_title, p1, p2, ncol = 1, rel_heights = c(0.1, 1, 1))

periods_train <- 12 * 100
periods_test  <- 12 * 50
skip_span     <- 12 * 22 - 1

rolling_origin_resamples <- rolling_origin(
  sun_spots,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples


# Plotting function for a single split
plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
    
    # Manipulate data
    train_tbl <- training(split) %>%
        add_column(key = "training") 
    
    test_tbl  <- testing(split) %>%
        add_column(key = "testing") 
    
    data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
        as_tbl_time(index = index) %>%
        mutate(key = fct_relevel(key, "training", "testing"))
        
    # Collect attributes
    train_time_summary <- train_tbl %>%
        tk_index() %>%
        tk_get_timeseries_summary()
    
    test_time_summary <- test_tbl %>%
        tk_index() %>%
        tk_get_timeseries_summary()
    
    # Visualize
    g <- data_manipulated %>%
        ggplot(aes(x = index, y = value, color = key)) +
        geom_line(size = size, alpha = alpha) +
        theme_tq(base_size = base_size) +
        scale_color_tq() +
        labs(
            title    = glue("Split: {split$id}"),
            subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
            y = "", x = ""
        ) +
        theme(legend.position = "none") 
    
    if (expand_y_axis) {
        
        sun_spots_time_summary <- sun_spots %>% 
            tk_index() %>% 
            tk_get_timeseries_summary()
        
        g <- g +
            scale_x_date(limits = c(sun_spots_time_summary$start, 
                                    sun_spots_time_summary$end))
    }
    
    return(g)
}

rolling_origin_resamples$splits[[1]] %>%
    plot_split(expand_y_axis = TRUE) +
    theme(legend.position = "bottom")

# Plotting function that scales to all splits 
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {
    
    # Map plot_split() to sampling_tbl
    sampling_tbl_with_plots <- sampling_tbl %>%
        mutate(gg_plots = map(splits, plot_split, 
                              expand_y_axis = expand_y_axis,
                              alpha = alpha, base_size = base_size))
    
    # Make plots with cowplot
    plot_list <- sampling_tbl_with_plots$gg_plots 
    
    p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
    legend <- get_legend(p_temp)
    
    p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
    
    p_title <- ggdraw() + 
        draw_label(title, size = 14, fontface = "bold", colour = palette_light()[[1]])
    
    g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
    
    return(g)
    
}

rolling_origin_resamples %>%
    plot_sampling_plan(expand_y_axis = T, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                       title = "Backtesting Strategy: Rolling Origin Sampling Plan")


rolling_origin_resamples %>%
    plot_sampling_plan(expand_y_axis = F, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                       title = "Backtesting Strategy: Zoomed In")


example_split    <- rolling_origin_resamples$splits[[6]]
example_split_id <- rolling_origin_resamples$id[[6]]

plot_split(example_split, expand_y_axis = FALSE, size = 0.5) +
    theme(legend.position = "bottom") +
    ggtitle(glue("Split: {example_split_id}"))

df_trn <- analysis(example_split)[1:800, , drop = FALSE]
df_val <- analysis(example_split)[801:1200, , drop = FALSE]
df_tst <- assessment(example_split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_val %>% add_column(key = "validation"),
  df_tst %>% add_column(key = "testing")
) %>%
  as_tbl_time(index = index)

df


rec_obj <- recipe(value ~ ., df) %>%
    step_sqrt(value) %>%
    step_center(value) %>%
    step_scale(value) %>%
    prep()

df_processed_tbl <- bake(rec_obj, df)

df_processed_tbl

center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history, "scale" = scale_history)


# model  ----------------------------------------------------------


FLAGS <- flags(
  # There is a so-called "stateful LSTM" in Keras. While LSTM is stateful per se,
  # this adds a further tweak where the hidden states get initialized with values 
  # from the item at same position in the previous batch.
  # This is helpful just under specific circumstances, or if you want to create an
  # "infinite stream" of states, in which case you'd use 1 as the batch size.
  # Below, we show how the code would have to be changed to use this, but it won't be further
  # discussed here.
  flag_boolean("stateful", FALSE),
  # Should we use several layers of LSTM?
  # Again, just included for completeness, it did not yield any superior performance on this task.
  # This will actually stack exactly one additional layer of LSTM units.
  flag_boolean("stack_layers", FALSE),
  # number of samples fed to the model in one go
  flag_integer("batch_size", 10),
  # size of the hidden state, equals size of predictions
  flag_integer("n_timesteps", 12),
  # how many epochs to train for
  flag_integer("n_epochs", 100),
  # fraction of the units to drop for the linear transformation of the inputs
  flag_numeric("dropout", 0.2),
  # fraction of the units to drop for the linear transformation of the recurrent state
  flag_numeric("recurrent_dropout", 0.2),
  # loss function. Found to work better for this specific case than mean squared error
  flag_string("loss", "logcosh"),
  # optimizer = stochastic gradient descent. Seemed to work better than adam or rmsprop here (as indicated by limited testing)
  flag_string("optimizer_type", "sgd"),
  # size of the LSTM layer
  flag_integer("n_units", 128),
  # learning rate
  flag_numeric("lr", 0.003),
  # momentum, an additional parameter to the SGD optimizer
  flag_numeric("momentum", 0.9),
  # parameter to the early stopping callback
  flag_integer("patience", 10)
)

# the number of predictions we'll make equals the length of the hidden state
n_predictions <- FLAGS$n_timesteps
# how many features = predictors we have
n_features <- 1
# just in case we wanted to try different optimizers, we could add here
optimizer <- switch(FLAGS$optimizer_type,
                    sgd = optimizer_sgd(lr = FLAGS$lr, momentum = FLAGS$momentum))

# callbacks to be passed to the fit() function
# We just use one here: we may stop before n_epochs if the loss on the validation set
# does not decrease (by a configurable amount, over a configurable time)
callbacks <- list(
  callback_early_stopping(patience = FLAGS$patience)
)


# functions used
build_matrix <- function(tseries, overall_timesteps) {
  t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x) tseries[x:(x + overall_timesteps - 1)]))
}

reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  X
}

# extract values from data frame
train_vals <- df_processed_tbl %>%
  filter(key == "training") %>%
  select(value) %>%
  pull()
valid_vals <- df_processed_tbl %>%
  filter(key == "validation") %>%
  select(value) %>%
  pull()
test_vals <- df_processed_tbl %>%
  filter(key == "testing") %>%
  select(value) %>%
  pull()

# build the windowed matrices
train_matrix <-
  build_matrix(train_vals, FLAGS$n_timesteps + n_predictions)
valid_matrix <-
  build_matrix(valid_vals, FLAGS$n_timesteps + n_predictions)
test_matrix <- build_matrix(test_vals, FLAGS$n_timesteps + n_predictions)

# separate matrices into training and testing parts
# also, discard last batch if there are fewer than batch_size samples (a purely technical requirement)
X_train <- train_matrix[, 1:FLAGS$n_timesteps]
y_train <- train_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
X_train <- X_train[1:(nrow(X_train) %/% FLAGS$batch_size * FLAGS$batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% FLAGS$batch_size * FLAGS$batch_size), ]

X_valid <- valid_matrix[, 1:FLAGS$n_timesteps]
y_valid <- valid_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
X_valid <- X_valid[1:(nrow(X_valid) %/% FLAGS$batch_size * FLAGS$batch_size), ]
y_valid <- y_valid[1:(nrow(y_valid) %/% FLAGS$batch_size * FLAGS$batch_size), ]

X_test <- test_matrix[, 1:FLAGS$n_timesteps]
y_test <- test_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
X_test <- X_test[1:(nrow(X_test) %/% FLAGS$batch_size * FLAGS$batch_size), ]
y_test <- y_test[1:(nrow(y_test) %/% FLAGS$batch_size * FLAGS$batch_size), ]

# add on the required third axis
X_train <- reshape_X_3d(X_train)
X_valid <- reshape_X_3d(X_valid)
X_test <- reshape_X_3d(X_test)

y_train <- reshape_X_3d(y_train)
y_valid <- reshape_X_3d(y_valid)
y_test <- reshape_X_3d(y_test)



model <- keras_model_sequential()

model %>%
  layer_lstm(
    units = FLAGS$n_units,
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
  compile(
    loss = FLAGS$loss,
    optimizer = optimizer,
    metrics = list("mean_squared_error")
  )

if (!FLAGS$stateful) {
  model %>% fit(
    x          = X_train,
    y          = y_train,
    validation_data = list(X_valid, y_valid),
    batch_size = FLAGS$batch_size,
    epochs     = FLAGS$n_epochs,
    callbacks = callbacks
  )
  
} else {
  for (i in 1:n_epochs) {
    model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_valid, y_valid),
      callbacks = callbacks,
      batch_size = FLAGS$batch_size,
      epochs     = 1,
      shuffle    = FALSE
    )
    model %>% reset_states()
  }
}

if (FLAGS$stateful)
  model %>% reset_states()


# create the model
model <- keras_model_sequential()

# add layers
# we have just two, the LSTM and the time_distributed 
model %>%
  layer_lstm(
    units = FLAGS$n_units, 
    # the first layer in a model needs to know the shape of the input data
    batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
    dropout = FLAGS$dropout,
    recurrent_dropout = FLAGS$recurrent_dropout,
    # by default, an LSTM just returns the final state
    return_sequences = TRUE
  ) %>% time_distributed(layer_dense(units = 1))

model %>%
  compile(
    loss = FLAGS$loss,
    optimizer = optimizer,
    # in addition to the loss, Keras will inform us about current MSE while training
    metrics = list("mean_squared_error")
  )

history <- model %>% fit(
  x          = X_train,
  y          = y_train,
  validation_data = list(X_valid, y_valid),
  batch_size = FLAGS$batch_size,
  epochs     = FLAGS$n_epochs,
  callbacks = callbacks
)

plot(history, metrics = "loss")

pred_train <- model %>%
  predict(X_train, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Retransform values to original scale
pred_train <- (pred_train * scale_history + center_history) ^2
compare_train <- df %>% filter(key == "training")

# build a dataframe that has both actual and predicted values
for (i in 1:nrow(pred_train)) {
  varname <- paste0("pred_train", i)
  compare_train <-
    mutate(compare_train,!!varname := c(
      rep(NA, FLAGS$n_timesteps + i - 1),
      pred_train[i,],
      rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
    ))
}

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

rsme_train

ggplot(compare_train, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_train1), color = "cyan") +
  geom_line(aes(y = pred_train50), color = "red") +
  geom_line(aes(y = pred_train100), color = "green") +
  geom_line(aes(y = pred_train150), color = "violet") +
  geom_line(aes(y = pred_train200), color = "cyan") +
  geom_line(aes(y = pred_train250), color = "red") +
  geom_line(aes(y = pred_train300), color = "red") +
  geom_line(aes(y = pred_train350), color = "green") +
  geom_line(aes(y = pred_train400), color = "cyan") +
  geom_line(aes(y = pred_train450), color = "red") +
  geom_line(aes(y = pred_train500), color = "green") +
  geom_line(aes(y = pred_train550), color = "violet") +
  geom_line(aes(y = pred_train600), color = "cyan") +
  geom_line(aes(y = pred_train650), color = "red") +
  geom_line(aes(y = pred_train700), color = "red") +
  geom_line(aes(y = pred_train750), color = "green") +
  ggtitle("Predictions on the training set")


pred_test <- model %>%
  predict(X_test, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Retransform values to original scale
pred_test <- (pred_test * scale_history + center_history) ^2
compare_test <- df %>% filter(key == "testing")

# build a dataframe that has both actual and predicted values
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

rsme_test

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
  geom_line(aes(y = pred_test550), color = "violet") +
  ggtitle("Predictions on test set")

obtain_predictions <- function(split) {
  df_trn <- analysis(split)[1:800, , drop = FALSE]
  df_val <- analysis(split)[801:1200, , drop = FALSE]
  df_tst <- assessment(split)
  
  df <- bind_rows(
    df_trn %>% add_column(key = "training"),
    df_val %>% add_column(key = "validation"),
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
    flag_integer("patience", 10)
  )
  
  n_predictions <- FLAGS$n_timesteps
  n_features <- 1
  
  optimizer <- switch(FLAGS$optimizer_type,
                      sgd = optimizer_sgd(lr = FLAGS$lr, momentum = FLAGS$momentum))
  callbacks <- list(
    callback_early_stopping(patience = FLAGS$patience)
  )
  
  train_vals <- df_processed_tbl %>%
    filter(key == "training") %>%
    select(value) %>%
    pull()
  valid_vals <- df_processed_tbl %>%
    filter(key == "validation") %>%
    select(value) %>%
    pull()
  test_vals <- df_processed_tbl %>%
    filter(key == "testing") %>%
    select(value) %>%
    pull()
  
  train_matrix <-
    build_matrix(train_vals, FLAGS$n_timesteps + n_predictions)
  valid_matrix <-
    build_matrix(valid_vals, FLAGS$n_timesteps + n_predictions)
  test_matrix <-
    build_matrix(test_vals, FLAGS$n_timesteps + n_predictions)
  
  X_train <- train_matrix[, 1:FLAGS$n_timesteps]
  y_train <-
    train_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_train <-
    X_train[1:(nrow(X_train) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_train <-
    y_train[1:(nrow(y_train) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_valid <- valid_matrix[, 1:FLAGS$n_timesteps]
  y_valid <-
    valid_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_valid <-
    X_valid[1:(nrow(X_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_valid <-
    y_valid[1:(nrow(y_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_test <- test_matrix[, 1:FLAGS$n_timesteps]
  y_test <-
    test_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_test <-
    X_test[1:(nrow(X_test) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_test <-
    y_test[1:(nrow(y_test) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_train <- reshape_X_3d(X_train)
  X_valid <- reshape_X_3d(X_valid)
  X_test <- reshape_X_3d(X_test)
  
  y_train <- reshape_X_3d(y_train)
  y_valid <- reshape_X_3d(y_valid)
  y_test <- reshape_X_3d(y_test)
  
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units            = FLAGS$n_units,
      batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
      dropout = FLAGS$dropout,
      recurrent_dropout = FLAGS$recurrent_dropout,
      return_sequences = TRUE
    )     %>% time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(
      loss = FLAGS$loss,
      optimizer = optimizer,
      metrics = list("mean_squared_error")
    )
  
  model %>% fit(
    x          = X_train,
    y          = y_train,
    validation_data = list(X_valid, y_valid),
    batch_size = FLAGS$batch_size,
    epochs     = FLAGS$n_epochs,
    callbacks = callbacks
  )
  
  
  pred_train <- model %>%
    predict(X_train, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # Retransform values
  pred_train <- (pred_train * scale_history + center_history) ^ 2
  compare_train <- df %>% filter(key == "training")
  
  for (i in 1:nrow(pred_train)) {
    varname <- paste0("pred_train", i)
    compare_train <-
      mutate(compare_train, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_train[i, ],
        rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  
  pred_test <- model %>%
    predict(X_test, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # Retransform values
  pred_test <- (pred_test * scale_history + center_history) ^ 2
  compare_test <- df %>% filter(key == "testing")
  
  for (i in 1:nrow(pred_test)) {
    varname <- paste0("pred_test", i)
    compare_test <-
      mutate(compare_test, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_test[i, ],
        rep(NA, nrow(compare_test) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  list(train = compare_train, test = compare_test)
  
}

all_split_preds <- rolling_origin_resamples %>%
  mutate(predict = map(splits, obtain_predictions))

calc_rmse <- function(df) {
  coln <- colnames(df)[4:ncol(df)]
  cols <- map(coln, quo(sym(.)))
  map_dbl(cols, function(col)
    rmse(
      df,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()
}


all_split_preds <- all_split_preds %>% unnest(predict)
all_split_preds_train <- all_split_preds[seq(1, 11, by = 2), ]
all_split_preds_test <- all_split_preds[seq(2, 12, by = 2), ]

all_split_rmses_train <- all_split_preds_train %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

all_split_rmses_test <- all_split_preds_test %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

all_split_rmses_train

all_split_rmses_test

plot_train <- function(slice, name) {
  ggplot(slice, aes(x = index, y = value)) + geom_line() +
    geom_line(aes(y = pred_train1), color = "cyan") +
    geom_line(aes(y = pred_train50), color = "red") +
    geom_line(aes(y = pred_train100), color = "green") +
    geom_line(aes(y = pred_train150), color = "violet") +
    geom_line(aes(y = pred_train200), color = "cyan") +
    geom_line(aes(y = pred_train250), color = "red") +
    geom_line(aes(y = pred_train300), color = "red") +
    geom_line(aes(y = pred_train350), color = "green") +
    geom_line(aes(y = pred_train400), color = "cyan") +
    geom_line(aes(y = pred_train450), color = "red") +
    geom_line(aes(y = pred_train500), color = "green") +
    geom_line(aes(y = pred_train550), color = "violet") +
    geom_line(aes(y = pred_train600), color = "cyan") +
    geom_line(aes(y = pred_train650), color = "red") +
    geom_line(aes(y = pred_train700), color = "red") +
    geom_line(aes(y = pred_train750), color = "green") +
    ggtitle(name)
}

train_plots <- map2(all_split_preds_train$predict, all_split_preds_train$id, plot_train)
p_body_train  <- plot_grid(plotlist = train_plots, ncol = 3)
p_title_train <- ggdraw() + 
  draw_label("Backtested Predictions: Training Sets", size = 18, fontface = "bold")

plot_grid(p_title_train, p_body_train, ncol = 1, rel_heights = c(0.05, 1, 0.05))

plot_test <- function(slice, name) {
  ggplot(slice, aes(x = index, y = value)) + geom_line() +
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
    geom_line(aes(y = pred_test550), color = "violet") +
    ggtitle(name)
}

test_plots <- map2(all_split_preds_test$predict, all_split_preds_test$id, plot_test)

p_body_test  <- plot_grid(plotlist = test_plots, ncol = 3)
p_title_test <- ggdraw() + 
  draw_label("Backtested Predictions: Test Sets", size = 18, fontface = "bold")

plot_grid(p_title_test, p_body_test, ncol = 1, rel_heights = c(0.05, 1, 0.05))

