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

library(purrr)



# Plotting function that scales to all splits
plot_sampling_plan <- function(sampling_tbl,
                               expand_y_axis = TRUE,
                               ncol = 3,
                               alpha = 1,
                               size = 1,
                               base_size = 14,
                               title = "Sampling Plan") {
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(
      gg_plots = map(
        splits,
        plot_split,
        expand_y_axis = expand_y_axis,
        alpha = alpha,
        base_size = base_size
      )
    )
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  p_title <- ggdraw() +
    draw_label(title,
               size = 18,
               fontface = "bold",
               colour = palette_light()[[1]])
  
  g <-
    plot_grid(p_title,
              p_body,
              legend,
              ncol = 1,
              rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}

calc_rmse <- function(prediction_tbl) {
  rmse_calculation <- function(data) {
    data %>%
      spread(key = key, value = value) %>%
      select(-index) %>%
      filter(!is.na(predict)) %>%
      rename(truth    = actual,
             estimate = predict) %>%
      rmse(truth, estimate)
  }
  
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
  
}
# Setup single plot function
plot_prediction <-
  function(data,
           id,
           alpha = 1,
           size = 2,
           base_size = 14) {
    rmse_val <- calc_rmse(data)
    
    g <- data %>%
      ggplot(aes(index, value, color = key)) +
      geom_point(alpha = alpha, size = size) +
      theme_tq(base_size = base_size) +
      scale_color_tq() +
      theme(legend.position = "none") +
      labs(
        title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
        x = "",
        y = ""
      )
    
    return(g)
  }

tidy_acf <- function(data, value, lags = 0:20) {
  value_expr <- enquo(value)
  
  acf_values <- data %>%
    pull(value) %>%
    acf(lag.max = tail(lags, 1), plot = FALSE) %>%
    .$acf %>%
    .[, , 1]
  
  ret <- tibble(acf = acf_values) %>%
    rowid_to_column(var = "lag") %>%
    mutate(lag = lag - 1) %>%
    filter(lag %in% lags)
  
  return(ret)
}

# Plotting function for a single split
plot_split <-
  function(split,
           expand_y_axis = TRUE,
           alpha = 1,
           size = 1,
           base_size = 14) {
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
        y = "",
        x = ""
      ) +
      theme(legend.position = "none")
    
    if (expand_y_axis) {
      sun_spots_time_summary <- sun_spots %>%
        tk_index() %>%
        tk_get_timeseries_summary()
      
      g <- g +
        scale_x_date(limits = c(
          sun_spots_time_summary$start,
          sun_spots_time_summary$end
        ))
    }
    
    return(g)
  }

plot_predictions <- function(sampling_tbl,
                             predictions_col,
                             ncol = 3,
                             alpha = 1,
                             size = 2,
                             base_size = 14,
                             title = "Backtested Predictions") {
  predictions_col_expr <- enquo(predictions_col)
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(
      gg_plots = map2(
        !!predictions_col_expr,
        id,
        .f        = plot_prediction,
        alpha     = alpha,
        size      = size,
        base_size = base_size
      )
    )
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  
  
  p_title <- ggdraw() +
    draw_label(title,
               size = 18,
               fontface = "bold",
               colour = palette_light()[[1]])
  
  g <-
    plot_grid(p_title,
              p_body,
              legend,
              ncol = 1,
              rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}

build_X <- function(tseries, lstm_num_timesteps) {
  X <- if (lstm_num_timesteps > 1) {
    t(sapply(1:(length(tseries) - lstm_num_timesteps),
             function(x)
               tseries[x:(x + lstm_num_timesteps - 1)]))
  } else {
    tseries[1:length(tseries) - lstm_num_timesteps]
  }
  if (lstm_num_timesteps == 1)
    dim(X) <- c(length(X), 1)
  cat("\nBuilt X matrix with dimensions: ", dim(X))
  return(X)
}

# get data into "timesteps form": target
build_y <- function(tseries, lstm_num_timesteps) {
  y <-
    sapply((lstm_num_timesteps + 1):(length(tseries)), function(x)
      tseries[x])
  cat("\nBuilt y vector with length: ", length(y))
  return(y)
}

reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  cat("\nReshaped X to dimensions: ", dim(X))
  return(X)
}

build_matrix <- function(tseries, overall_timesteps) {
  X <-
    t(sapply(1:(length(tseries) - overall_timesteps + 1), #!!!!!!!!!! +1
             function(x)
               tseries[x:(x + overall_timesteps - 1)]))
  cat("\nBuilt matrix with dimensions: ", dim(X))
  return(X)
}

calc_multiple_rmse <- function(df) {
  m <- as.matrix(df)
  ground_truth <- m[, 1]
  pred_cols <- m[, 2:ncol(df)]
  rowwise_rsme <-
    apply(pred_cols, 2, function(col)
      sqrt(sum((col - ground_truth) ^ 2  / length(na.omit(col)), na.rm = TRUE
      )))
  mean(rowwise_rsme)
}

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
