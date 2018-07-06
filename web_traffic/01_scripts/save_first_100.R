# SAVE FIRST 100 WEB SITES

library(data.table)
library(tidyverse)

train_2_raw <- fread("00_data/train_2.csv") %>%
    as.tibble() %>%
    rowid_to_column(var = "rowid")

train_2_raw_100 <- train_2_raw %>%
    slice(1:100)

write.csv(train_2_raw_100, "00_data/train_2_raw_100.csv", row.names = F)
