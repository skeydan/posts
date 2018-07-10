library(keras)

n_tsteps <- 3:10

# [1] 3
# 9.069388 
# 9.071607 
# [1] 4
# 9.161394 
# 9.161016 
# [1] 5
# 8.111531 
# 8.116871 
# [1] 6
# 6.384771 
# 6.384593 
# [1] 7
# 2.210652 
# 2.211505 
# [1] 8
# 6.486989 
# 6.489145 
# [1] 9
# 8.685701 
# 8.680563 
# [1] 10
# 9.642423 
# 9.634106
  
batch_size <- 10
n_epochs <- 200

data <- rep(1:7, 10) + rnorm(70) %>% round(digits = 2)

for (t in seq_along(n_tsteps)) {
  print(n_tsteps[t])
  
  data_matrix <-
    sapply(1:(length(data) - n_tsteps[t] * 2 + 1), function(index) {
      data[index:(index + n_tsteps[t] * 2 - 1)]
    }) %>% t()
  data_matrix
  x <- data_matrix[, 1:n_tsteps[t]]
  y <- data_matrix[, (n_tsteps[t] + 1):ncol(data_matrix)]
  dim(x) <- c(dim(x)[1], dim(x)[2], 1)
  dim(y) <- c(dim(y)[1], dim(y)[2], 1)
  
  
  set.seed(7777)
  model1 <- keras_model_sequential()
  model1 %>% time_distributed(layer_dense(units = 1, kernel_initializer = initializer_zeros(), input_shape = c(n_tsteps[t], 1)))
  model1 %>% compile(loss = "mse", optimizer = "adam")
  history <-
    model1 %>% fit(x,
                   y,
                   epochs = n_epochs,
                   batch_size = batch_size,
                   verbose = 0)
  plot(history)
  model1 %>% print()
  model1$get_weights()  %>% print()
  model1 %>% evaluate(x, y) %>% print()
  
  set.seed(7777)
  model2 <- keras_model_sequential()
  model2 %>% layer_dense(units = 1, input_shape = c(n_tsteps[t], 1), kernel_initializer = initializer_zeros())
  model2 %>% print()
  model2 %>% compile(loss = "mse", optimizer = "adam")
  history <-
    model2 %>% fit(x,
                   y,
                   epochs = n_epochs,
                   batch_size = batch_size,
                   verbose = 0)
  plot(history)
  model2$get_weights()  %>%  print()
  model2 %>% evaluate(x, y) %>% print()
  
}
