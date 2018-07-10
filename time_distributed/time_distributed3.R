library(keras)

n_tsteps <- 3:10
n_features <- 2
batch_size <- 10
n_epochs <- 200

col1 <- rep(1:7, 10) + rnorm(70) %>% round(digits = 2)
col2 <- rep(1:70) + rnorm(70) %>% round(digits = 2)

for (t in seq_along(n_tsteps)) {
  print(n_tsteps[t])
  
  col1_matrix <-
    sapply(1:(length(col1) - n_tsteps * 2 + 1), function(index) {
      col1[index:(index + n_tsteps * 2 - 1)]
    }) %>% t()
  col1_matrix
  
  col2_matrix <-
    sapply(1:(length(col2) - n_tsteps * 2 + 1), function(index) {
      col2[index:(index + n_tsteps * 2 - 1)]
    }) %>% t()
  col2_matrix
  
  data_array <-
    array(0, dim = c(nrow(col1_matrix), n_tsteps * 2, n_features))
  data_array[, , 1] <- col1_matrix
  data_array[, , 2] <- col2_matrix
  
  x <- data_array[, 1:n_tsteps,]
  y <- data_array[, (n_tsteps + 1):ncol(data_array),]
  
  set.seed(7777)
  model1 <- keras_model_sequential()
  model1 %>% time_distributed(layer_dense(units = n_features,
                                          kernel_initializer = initializer_zeros()),
                              input_shape = c(n_tsteps[t], n_features))
  model1
  model1 %>% compile(loss = "mse", optimizer = "adam")
  history <-
    model1 %>% fit(x,
                   y,
                   epochs = n_epochs,
                   batch_size = batch_size,
                   verbose = 0)
  plot(history)
  model1 %>% evaluate(x, y) %>% print()
  
  m1weights <- model1$get_weights()[[1]]
  m1weights
  m1bias <- model1$get_weights()[[2]]
  m1bias
  
  
  model1 %>% predict(x[1, , , drop = FALSE]) %>% .[, , 1]
  model1 %>% predict(x[1, , , drop = FALSE]) %>% .[, , 2]
  xr1 <- x[1, ,] %>% t()
  xr1
  pred1 <- m1weights %*% xr1
  pred1
  m1bias_bc <- matrix(0, nrow = 2, ncol = 6)
  m1bias_bc[,] <- m1bias
  m1bias_bc
  pred1 + m1bias_bc
  
  # this worked
  # > pred1 <- xr1 %*% m1weights
  # > pred1 <- xr1 %*% m1weights %>% .[1, ]
  # > pred1
  # [1] -0.04405802  2.28569818
  # > pred1 + m1bias
  # [1] 0.543916 2.809396
  # > model1 %>% predict(x[1, , , drop = FALSE]) %>% .[ , , 1]
  # [1] 0.543916 1.179607 1.270997 1.815340 1.870294 2.331417
  # > model1 %>% predict(x[1, , , drop = FALSE]) %>% .[ , , 2]
  # [1] 2.809396 3.712628 4.169200 5.902379 8.185821 8.371355
  #pred1 <- xr1 %*% m1weights
  
  
  set.seed(7777)
  model2 <- keras_model_sequential()
  model2 %>% layer_dense(units = n_features,
                         input_shape = c(n_tsteps, n_features),
                         kernel_initializer = initializer_zeros())
  model2
  model2 %>% compile(loss = "mse", optimizer = "adam")
  history <-
    model2 %>% fit(x,
                   y,
                   epochs = n_epochs,
                   batch_size = batch_size,
                   verbose = 0)
  plot(history)
  model2$get_weights()
  model2 %>% evaluate(x, y)
  x[1, , 1]
  model2 %>% predict(x[1, , , drop = FALSE]) %>% .[, , 1]
  x[1, , 2]
  model2 %>% predict(x[1, , , drop = FALSE]) %>% .[, , 2]
}
