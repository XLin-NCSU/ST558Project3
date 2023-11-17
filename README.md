# ST558 Project 3

## Purpose of the repo
This repo creates predictive models using available data and automates R Markdown reports. We demonstrated it by using [Diabates Health Indicators Data](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) collected by the Centers of Disease Control and Prevention (CDC).  

## List of R Packages used for the Project
```{r}
library(tidyverse)
library(corrplot)
library(visreg)
library(caret)
```

## Render Code 
Here is the code we used to create analyses from a single .Rmd file.

```{r}
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
```

## Links to Output Files
The output files of this analysis can be found at following links:
  For example,
  
  - Analysis for \[College Graduates\]\(college_graduate_analysis.html\). Note you should only
have a college_graduate_analysis.md file in the repo - github pages will render the .html file
for you
