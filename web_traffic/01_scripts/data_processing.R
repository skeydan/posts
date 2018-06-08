# DATA PROCESSING ----
# Functions to help work with the time series

library(tidyverse)
library(tidyquant)
library(padr)

extract_timeseries_by_id <- function(data, id, pad_to = "2017-09-10") {
    
    data %>%
        filter(rowid %in% id) %>%
        gather(key = date, value = value, -c(rowid, Page)) %>%
        mutate(date  = as_date(date)) %>%
        group_by(rowid, Page) %>%
        filter(!is.na(value)) %>%
        pad(interval = "day", end_val = ymd(pad_to)) %>%
        fill_by_value(value = 0) %>%
        tbl_time(index = date)
    
}

plot_timeseries <- function(data, ncol = 1) {
    
    n_rowid <- data %>%
        ungroup() %>%
        pull(rowid) %>%
        unique() %>%
        length()
    
    data %>%
        ungroup() %>%
        mutate(
            rowid_text = paste0("Row ID: ", rowid) %>% fct_reorder(rowid),
            rowid      = as.factor(rowid)
            ) %>%
        ggplot(aes(date, value, color = rowid)) +
        geom_line() +
        facet_wrap(~ rowid_text, ncol = ncol, scales = "free_y") +
        theme_tq() +
        scale_color_manual(values = rep(palette_light(), n_rowid)) +
        labs(x = "Date", y = "Web Traffic") +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
            legend.position = "none"
        )
    
}
