library(keras)

n_tsteps <- 3:10

# [1] 3
# 19.41928 
# 18.89665 
# [1] 4
# 20.82441 
# 20.29704 
# [1] 5
# 25.82348 
# 24.88567 
# [1] 6
# 41.90795 
# 41.74705 
# [1] 7
# 43.9716 
# 44.00468 
# [1] 8
# 52.03735 
# 52.77912 
# [1] 9
# 59.93825 
# 59.16297 
# [1] 10
# 70.20051 
# 69.7509 

  
n_features <- 2
batch_size <- 10
n_epochs <- 200

col1 <- rep(1:7, 10) + rnorm(70) %>% round(digits = 2)
col2 <- rep(1:70) + rnorm(70) %>% round(digits = 2)

for (t in seq_along(n_tsteps)) {
  print(n_tsteps[t])
  
  col1_matrix <-
    sapply(1:(length(col1) - n_tsteps[t] * 2 + 1), function(index) {
      col1[index:(index + n_tsteps[t] * 2 - 1)]
    }) %>% t()
  col1_matrix
  
  col2_matrix <-
    sapply(1:(length(col2) - n_tsteps[t] * 2 + 1), function(index) {
      col2[index:(index + n_tsteps[t] * 2 - 1)]
    }) %>% t()
  col2_matrix
  
  data_array <-
    array(0, dim = c(nrow(col1_matrix), n_tsteps[t] * 2, n_features))
  data_array[, , 1] <- col1_matrix
  data_array[, , 2] <- col2_matrix
  
  x <- data_array[, 1:n_tsteps[t],]
  y <- data_array[, (n_tsteps[t] + 1):ncol(data_array),]
  
  set.seed(7777)
  model1 <- keras_model_sequential()
  model1 %>% time_distributed(layer_dense(units = n_features,
                                          kernel_initializer = initializer_zeros()),
                              input_shape = c(n_tsteps[t], n_features))
  model1 %>% print()
  model1 %>% compile(loss = "mse", optimizer = "adam")
  history <-
    model1 %>% fit(x,
                   y,
                   epochs = n_epochs,
                   batch_size = batch_size,
                   verbose = 0)
  plot(history)
  model1$get_weights()  %>%  print()
  model1 %>% evaluate(x, y) %>% print()
 
  set.seed(7777)
  model2 <- keras_model_sequential()
  model2 %>% layer_dense(units = n_features,
                         input_shape = c(n_tsteps[t], n_features),
                         kernel_initializer = initializer_zeros())
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
