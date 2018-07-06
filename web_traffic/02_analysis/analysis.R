

library(tidyverse)
library(tidyquant)
library(padr)
library(tibbletime)

source("01_scripts/data_processing.R")

# Collect Data
train_2_raw <- read_csv("00_data/train_2_raw_100.csv")

train_2_raw %>%
    select(1:7, ncol(.))

# Visualize Data

train_2_raw %>%
    extract_timeseries_by_id(1:20, pad_to = "2017-09-10") %>%
    plot_timeseries(ncol = 4) +
    labs(title = "First 20 Observations")

    
train_2_raw %>%
    extract_timeseries_by_id(80:100, pad_to = "2017-09-10") %>%
    plot_timeseries(ncol = 4) +
    labs(title = "Last 20 Observations")

# Feature exploration

train_2_raw$Page[[1]]


