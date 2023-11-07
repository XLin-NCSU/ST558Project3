library(rmarkdown)
library(tidyverse)

Education_Group <- c(2,3,4,5,6)

output_file <- paste0("Education Group ",Education_Group,".md")

params = lapply(Education_Group, FUN = function(x){list(Education = x)})

reports <- tibble(output_file, params)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Project3.Rmd", output_file = x[[1]], params = x[[2]], github_document(html_preview = FALSE))
      })
