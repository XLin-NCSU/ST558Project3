# ST558 Project 3

## a brief description of the purpose of the repo
  
## a list of R packages used
```{r}
library(rmarkdown)
library(tidyverse)
library(caret)
```
## the code used to create the analyses from a single .Rmd file (i.e. the render() code)

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

## links to .html files of the generated analyses (which will be created by github pages! Not you!)

  For example,
  
  - Analysis for \[College Graduates\]\(college_graduate_analysis.html\). Note you should only
have a college_graduate_analysis.md file in the repo - github pages will render the .html file
for you
