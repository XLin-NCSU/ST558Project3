Project 3
================
Xi Lin, Sarat Bantupalli

## Required Packages

## Introduction

This project creates predictive models using available data and
automates R Markdown reports. We demonstrated it by using [Diabates
Health Indicators
Data](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
collected by the Centers of Disease Control and Prevention (CDC).

The data was collected through a telephonic survey by the CDC in the
year 2015 and it corresponds to the prevalence of diabetes in the survey
respondents. The data set we analyzed here represents responses from
253,680 Americans on health-related risk behaviors, chronic health
conditions, and the use of preventative services. The data set itself is
*not balanced*. For this reason, rather than depending on accuracy (a
step function) as the metric to evaluate different models, a continuous
function (LogLoss) was used.

### Model Variables

The response variable of interest from the data set is
*Diabetes_binary*. It represents if the survey respondent is *non
diabetic (value 0)* or *pre-diabetic/ diabetic (value 1)*.

The explanatory variables were chosen such that important risk factors
for diabetes including **health & lifestyle, socioeconomic status, and
demographics (age and sex)** are considered in the predictive model. We
included the following explanatory variables for the model:

- *HighBP*, a health factor: This variable represents if a respondent
  was diagnosed with high blood pressure (value 1) or not (value 0).
  Research indicates high blood pressure is twice as likely to strike a
  person with diabetes than a person without diabetes  
- *HighChol*, a health factor: This variable represents if a respondent
  has high cholesterol (value 1) or not (value 0). Research suggests
  people with diabetes tend to have higher cholesterol levels  
- *BMI*, a health factor: The body mass index, a numerical variable, is
  a good indicator for obesity and represents the overall health of a
  person making it an ideal candidate for the model  
- *Smoker*, a lifestyle factor: Smoking increases the risk of diabetes
  and hence was included in the model. This variable can take two
  values, 0- did not smoke 100 cigarettes in lifetime or 1- smoked more
  than 100 cigarettes in lifetime  
- *PhysicalActivity*, a lifestyle factor: This variable indicates if the
  respondent had been physically active in the past 30 days. A value 0
  represents no while 1 represents yes  
- *Fruits*, a lifestyle or socioeconomic factor: A healthy diet is key
  to reducing the risk of diabetes and can represent lifestyle and
  socioeconomic status of the respondent. This variable indicates if the
  person is eating healthy or not. A value of 0 indicates no fruit
  consumed per day while 1 indicates some fruit consumed per day  
- *Veggies*, a lifestyle or socioeconomic factor: This variable
  indicates if the person is eating healthy or not. It can represent
  lifestyle and socioeconomic status of the respondent. A value of 0
  indicates no veggies consumed per day while 1 indicates some veggies
  consumed per day  
- *HvyAlcoholConsump*, a lifestyle factor: Alcohol increases the risk of
  diabetes and this variable encapsulates if the respondent is a heavy
  drinker (value 1) or not (value 0)  
- *AnyHealthcare*, a socioeconomic factor: This variable represents
  socioeconomic status of the respondent, with low-income respondents
  not having any health coverage (value 0) while higher income
  respondents having some type of health coverage (value 1)  
- *GenHlth*, a health factor: This represents the overall health of the
  respondent ranging from 1 through 5 with 1 being the best health
  indicator  
- *MentHlth*, a health or socioeconomic factor: This numerical variable
  represents days of poor mental health (1-30 days). People with mental
  health are 2 to 3 times more likely to have a depression than people
  without diabetes. This variable captures the overall health of the
  respondent  
- *PhysHlth*, a health factor: This numerical variable indicates
  physical illness or injury days in the past 30 days. Illness or injury
  can impact physical activity which in turn could affect diabetes
  risk  
- *Sex*, a demographic factor: Including the sex of the respondent (0-
  female, 1- male) could help us see if risk factors are different for
  men and women  
- *Age*, a demographic factor: Risk for type 2 diabetes increases with
  age. This 13 level categorical variable encapsulates this risk  
- *Education*, a socioeconomic factor: Education might effect the
  socioeconomic status of a person and change the risk factor for
  diabetes. This variable was used to automate the R Markdown reports.
  Each report represents analysis for each level of this variable. A
  value of 2 indicates the respondent attended elementary school or less
  while 6 represents a college graduate  
- *Income*, a socioeconomic factor: This categorical variable can affect
  factors that influence the risk for type 2 diabetes including access
  to health care, physical activity (eg. gym), and a healthy lifestyle.
  A value of 1 indicates low income (earning less than \$10,000), and a
  value of 8 indicates high income (\>\$75,000)

### Purpose of EDA and Modeling

We used this project to develop predictive models for diabetes risk.
With over 110 million Americans who are either diabetic or pre-diabetic,
predictive models such as these can be of immense help for public health
officials. We looked at how health & lifestyle, socioeconomic status,
and demographics can help us predict the risk of diabetes for a person
living in the United States. We used the power of R Markdown to automate
the report generation process based on the *Education* level of the
person.

Exploratory data analysis was the first step of this predictive modeling
project. Although the variable selection process was based on knowledge
of the field and metadata of the data set, EDA played an important role.
EDA helped us visualize the data, look for trends, and see correlation
between variables. Using EDA we were able to quantify (through summaries
and tables) / visualize (through plots) how different variables affect
the risk for diabetes.

Through this project we want to see how a confluence of factors
including health & lifestyle, socioeconomic status, and demographics of
an individual can help us predict diabetes risk. We developed several
linear and non-linear models and chose the best model based on the
logloss value of the models.

## Data Processing

The data for the analysis was read in using the `read_csv` function from
the `readr` package and stored in the R object, *diabetes_data*. Then
for the variable *Education*, groups 1 and 2 were combined into *group
2*, combining respondents with elementary school education or less into
one group. This was done to have five distinct groups for this variable
ranging from 2 through 6.

Then the response variable and some of he explanatory variables of
interest were converted to factors. Explanatory variables that were
converted to factors with meaningful level names include *HighBP*,
*HighChol*, *Smoker*, *PhysicalActivity*, *Fruits*, *Veggies*,
*HvyAlcoholConsump*, *AnyHealthcare*, *GenHlth*, *Sex*, *Age*, and
*Income*.

The data was then subset to represent different *Education* levels of
the respondents in the survey.

``` r
#Read in data using the read_csv function from readr package
diabetes_data <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

#Combine groups 1 and 2 into group 2 for Education variable
diabetes_data <- diabetes_data %>% 
  mutate(Education = if_else(Education == 1, 2, Education) %>% as.factor())

# Re-name the Diabetes_binary variable to a more meaningful description
diabetes_data$Diabetes_binary <- if_else(diabetes_data$Diabetes_binary == 0, 
                                    "No_Diabetes","Diabetes")

# Select only the variables we are interested in
diabetes_data <- diabetes_data %>% 
  select(Diabetes_binary, HighBP, HighChol, BMI, Smoker, PhysActivity,
         Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, GenHlth,
         MentHlth, PhysHlth, Sex, Age, Education, Income)

# Create a vector with the columns we want to convert to factors
columns_to_factor <- c("Diabetes_binary",
                       "HighBP",
                       "HighChol",
                       "Smoker",
                       "PhysActivity",
                       "Fruits",
                       "Veggies",
                       "HvyAlcoholConsump",
                       "AnyHealthcare",
                       "GenHlth",
                       "Sex",
                       "Age",
                       "Income")

# Change the columns we are interested in to a factor using the lapply function
diabetes_data[, columns_to_factor] <- lapply(diabetes_data[, columns_to_factor], factor)

# Subset the data to work on the Education level of interest
diabetes_education_level <- subset(diabetes_data, Education == params$Education) %>% 
  select(-Education)

#diabetes_education_level <- subset(diabetes_data, Education == 2) %>% 
  #select(-Education)
```

## Summarizations and Exploratory Data Analysis (EDA)

Here Exploratory Data Analysis on the full data (for the particular
education level) was performed to look at trends in data, and
correlations between variables.

### Questions

The questions we are interested in exploring from the diabetes data are:

- *affects of health & lifestyle on diabetes risk*  
- *affects of socioeconomic status on diabetes risk*  
- *affects of demographics on diabetes risk*

### Is the Data Balanced or Not!!

An important measure to check prior to any analysis is the balance of
data. If the data is unbalanced, predictive models will be biased
towards one level of the response variable. In case of unbalanced data
sets, model metrics like *logloss* value should be considered.

Using the `summary` function to look at the balance of data for
Education level

``` r
# Check for the balance of data 
summary(diabetes_education_level$Diabetes_binary)
```

    ##    Diabetes No_Diabetes 
    ##        1230        2987

### Health & Lifestyle on Diabetes Risk

Here we looked at the influence of *Health and Lifestyle* variables on
diabetes risk. The summary statistics of diabetes_status with respect to
*BMI*, *Mental Health*, and *Physical Health* were found using the
`group_by()` function. By comparing the mean response of these
predictors, we can get an overall picture of how health variables
influence diabetes risk on average. The trends can help in guiding the
research questions.

``` r
# # Summary of BMI, Mental Health, and Physical Health for 
# diabetes vs non-diabetes respondents in the survey
diabetes_education_level %>% group_by(Diabetes_binary) %>% 
  summarise(BMI_avg = mean(BMI), BMI_sd = sd(BMI),
            MentHlth_avg = mean(MentHlth), MentHlth_sd = sd(MentHlth),
            PhysHlth_avg = mean(PhysHlth), PhysHlth_sd = sd(PhysHlth))
```

    ## # A tibble: 2 × 7
    ##   Diabetes_binary BMI_avg BMI_sd MentHlth_avg MentHlth_sd PhysHlth_avg PhysHlth_sd
    ##   <fct>             <dbl>  <dbl>        <dbl>       <dbl>        <dbl>       <dbl>
    ## 1 Diabetes           31.4   7.40         6.67       10.8         11.3         12.5
    ## 2 No_Diabetes        28.7   6.75         4.62        9.27         7.18        11.0

The reader should look for the mean value of predictors at different
settings of diabetes_status. For people without diabetes, on average a
lower value of BMI, lower days of mental health issues, and lower days
of physical injury is expected.

Here we looked at how *lifestyle indicators* affect diabetes status for
an individual. We considered *smoking status*, *eating habits (fruits
and veggies)*, and *alcohol consumption* for this part.

``` r
# Summary proportion of how lifestyle impacts diabetes status
lifestyle_table <- diabetes_education_level %>% 
  group_by(Diabetes_binary, Fruits, Veggies, HvyAlcoholConsump, Smoker) %>% 
  summarise(n = n()) %>% mutate(freq = n/sum(n))

lifestyle_table
```

    ## # A tibble: 31 × 7
    ## # Groups:   Diabetes_binary, Fruits, Veggies, HvyAlcoholConsump [16]
    ##    Diabetes_binary Fruits Veggies HvyAlcoholConsump Smoker     n  freq
    ##    <fct>           <fct>  <fct>   <fct>             <fct>  <int> <dbl>
    ##  1 Diabetes        0      0       0                 0        101 0.406
    ##  2 Diabetes        0      0       0                 1        148 0.594
    ##  3 Diabetes        0      0       1                 0          1 0.25 
    ##  4 Diabetes        0      0       1                 1          3 0.75 
    ##  5 Diabetes        0      1       0                 0        133 0.470
    ##  6 Diabetes        0      1       0                 1        150 0.530
    ##  7 Diabetes        0      1       1                 1          2 1    
    ##  8 Diabetes        1      0       0                 0         75 0.5  
    ##  9 Diabetes        1      0       0                 1         75 0.5  
    ## 10 Diabetes        1      0       1                 0          2 0.667
    ## # ℹ 21 more rows

The reader should look at trends in proportion of people with diabetes
who lead a healthy lifestyle versus those who do not. For example
proportion of people with diabetes who eat healthy, do not smoke, and do
not consume alcohol is `{r} lifestyle_table$freq[1]` while proportion of
people with diabetes who eat healthy, do not smoke but consume alcohol
is `{r} lifestyle_table$freq[2]`. This table summarizes how different
lifestyles can change the risk for diabetes.

Below, we visualized the effect of health and lifestyle on diabetes
risk. For graphics purposes, we assumed a logistic regression model for
the response variable (Diabetes_binary) with only the main effects of
all variables and plotted how risk of diabetes changes with respect to
BMI.

``` r
plot_glm <- glm( Diabetes_binary ~ .,
            family = "binomial", data = diabetes_education_level)

visreg(plot_glm, "BMI", gg = TRUE, scale = "response") + 
  labs(y = "Prob(Not having Diabetes)", x = "BMI", 
       title = "Probability(Not having Diabetes or Being Healthy) vs BMI") +
  theme_bw()
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
The y-axis of the plot displays the probability of being healthy/ not
having diabetes. The reader should look for trends on how the BMI of an
individual changes the risk of diabetes.

Here, we created a visual of the correlation among different health
indicating variables using the *corrplot* library.

``` r
# Correlation of numeric variables.
Correlation <- cor(select(diabetes_education_level, 
                          c("BMI","MentHlth","PhysHlth")))
corrplot(Correlation,  tl.pos = "lt")
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

The reader should look for any strong (positive or negative) correlation
among the numeric variables to minimize collinearity.

### Socioeconomic Status on Diabetes Risk

**Socioeconomic** status plays an important role in risk for diabetes.
Here we summarized the impact of *Income* and *AnyHealthcare* on
diabetes_status.

``` r
socioeconomic_table <- diabetes_education_level %>% 
  group_by(Diabetes_binary, Income, AnyHealthcare) %>% 
  summarise(n = n()) %>% mutate(freq = n/sum(n))

socioeconomic_table
```

    ## # A tibble: 32 × 5
    ## # Groups:   Diabetes_binary, Income [16]
    ##    Diabetes_binary Income AnyHealthcare     n   freq
    ##    <fct>           <fct>  <fct>         <int>  <dbl>
    ##  1 Diabetes        1      0                25 0.0716
    ##  2 Diabetes        1      1               324 0.928 
    ##  3 Diabetes        2      0                25 0.0899
    ##  4 Diabetes        2      1               253 0.910 
    ##  5 Diabetes        3      0                30 0.139 
    ##  6 Diabetes        3      1               186 0.861 
    ##  7 Diabetes        4      0                22 0.147 
    ##  8 Diabetes        4      1               128 0.853 
    ##  9 Diabetes        5      0                15 0.118 
    ## 10 Diabetes        5      1               112 0.882 
    ## # ℹ 22 more rows

The reader should look for trends on how Income and accessibility of
health care plays a role in diabetes_status. For example proportion of
people with diabetes who are wealthy and have health care is
`{r} socioeconomic_table$freq[16]` while proportion of people with
diabetes who are not wealthy and have health care is
`{r} socioeconomic_table$freq[2]`.

We also wanted to look for trends in diabetes and socioeconomic status,
especially *Income*.

``` r
visreg(plot_glm, "Income", gg = TRUE, scale = "response") + 
  labs(y = "Prob(Not having Diabetes)", x = "Income Level", 
       title = "Probability(Not having Diabetes or Being Healthy) vs BMI") +
  theme_bw()
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->
The y-axis of the plot displays the probability of being healthy/ not
having diabetes. An interesting trend to observe here will be the risk
of diabetes with income. Since diabetes risk is expected to be higher
for low-income families, this plot gives an important insight.

### Demographics on Diabetes Risk

**Demographics (age and sex)** can have a significant role in the risk
for diabetes. Here we summarized the impact of *Age* and *Sex* on
diabetes_status.

``` r
demographics_table <- diabetes_education_level %>% group_by(Diabetes_binary, Age, Sex) %>% 
  summarise(n = n()) %>% mutate(freq = n/sum(n))

demographics_table
```

    ## # A tibble: 51 × 5
    ## # Groups:   Diabetes_binary, Age [26]
    ##    Diabetes_binary Age   Sex       n  freq
    ##    <fct>           <fct> <fct> <int> <dbl>
    ##  1 Diabetes        1     1         1 1    
    ##  2 Diabetes        2     0         3 0.75 
    ##  3 Diabetes        2     1         1 0.25 
    ##  4 Diabetes        3     0         7 0.778
    ##  5 Diabetes        3     1         2 0.222
    ##  6 Diabetes        4     0         5 0.357
    ##  7 Diabetes        4     1         9 0.643
    ##  8 Diabetes        5     0        12 0.571
    ##  9 Diabetes        5     1         9 0.429
    ## 10 Diabetes        6     0        40 0.625
    ## # ℹ 41 more rows

The reader should simultaneously look at trends for age and sex in this
table.

Here we looked at the impact of **demographics (age and sex)** on
diabetes_status using a 3-way contingency table.

``` r
# 3-way contingency table for diabetes status, age and sex
table(diabetes_education_level$Age, diabetes_education_level$Sex,
      diabetes_education_level$Diabetes_binary)
```

    ## , ,  = Diabetes
    ## 
    ##     
    ##        0   1
    ##   1    0   1
    ##   2    3   1
    ##   3    7   2
    ##   4    5   9
    ##   5   12   9
    ##   6   40  24
    ##   7   53  32
    ##   8   81  47
    ##   9   98  65
    ##   10 120  83
    ##   11  89 104
    ##   12  91  70
    ##   13 108  76
    ## 
    ## , ,  = No_Diabetes
    ## 
    ##     
    ##        0   1
    ##   1   10  15
    ##   2   33  15
    ##   3   56  48
    ##   4   78  79
    ##   5  112  88
    ##   6  108  95
    ##   7  163 145
    ##   8  138 135
    ##   9  150 141
    ##   10 171 149
    ##   11 164 151
    ##   12 157 128
    ##   13 236 222

The reader should look for differences in male and female
diabetes_status and also how age impacts it!!

Below, we visualized the effect of demographics on diabetes risk. For
graphics purposes, we assumed a logistic regression model for the
response variable (Diabetes_binary) with only the main effects of all
variables and plotted how risk of diabetes changes with respect to age
and sex.

``` r
plot_data <- diabetes_education_level
plot_data$Sex <- if_else(plot_data$Sex == 0, "Female","Male")

plot_glm1 <- glm( Diabetes_binary ~ .,
            family = "binomial", data = plot_data)

visreg(plot_glm1, "Age", by = "Sex",gg = TRUE, scale = "response") + 
  labs(y = "Prob(Not having Diabetes)", x = "Age", 
       title = "Probability(Not having Diabetes) vs Age") 
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->
The reader should look for trends on how the age group and sex of an
individual changes the risk of diabetes.

Here without any assumptions (of model fits), we looked at trends on the
influence of age on diabetes_status using a barplot.

``` r
# Barplot of respondents age and diabetes status
g1 <- ggplot(data = diabetes_education_level, aes(x = Age)) +
  labs(x = "Age Group", title = "Bar Plot of Age of Respondents in the diabetes study",
       y = "number of respondents") + 
  scale_fill_discrete(name = "Diabetes Status", labels = c("Have diabetes", "Don't have diabetes"))
g1 + geom_bar(aes(fill = Diabetes_binary)) + theme_bw()
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

The reader should look for trends in how the proportion of people with
diabetes changes with age.

We also looked at trends in diabetes_status for men and women.

``` r
# Barplot of respondents age and diabetes status
g2 <- ggplot(data = diabetes_education_level, aes(x = Sex)) +
  labs(x = "Sex", title = "Bar Plot of Sex of Respondents in the diabetes study",
       y = "number of respondents") + 
  scale_fill_discrete(name = "Diabetes Status", labels = c("Have diabetes", "Don't have diabetes")) +
  scale_x_discrete(labels = c("Female", "Male"))
g2 + geom_bar(aes(fill = Diabetes_binary)) + theme_bw()
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
The reader should look for trends in how diabetes_status changes for men
and women.

## Modeling

### Splitting Train and Test data sets

Here the data was split into training set (70% of data) and test set
(30% of data). We have used the `set.seed` function to have reproducible
results.

``` r
set.seed(12345)
# Create training data using createDataPartition function from caret package

train_data_index <- createDataPartition(diabetes_education_level$Diabetes_binary, p = 0.7, 
                                        list = FALSE)
diabetes_train <- diabetes_education_level[train_data_index, ]
diabetes_test <- diabetes_education_level[-train_data_index, ]
```

### Log-loss

Log-loss, also known as cross-entropy loss or logistic loss, is a
*common evaluation metric* for binary classification problems. It is
indicative of how close the prediction probability is to the actual
value. A *lower log-loss* value indicates a lower deviance in predicted
probability vs observed probability and is desired.

Log-loss is a more informative metric than other metrics like accuracy
for several reasons:

- It **penalizes models** for being confident in their wrong
  predictions. Accuracy simply counts the number of correct predictions,
  but it does not consider how confident the model was in those
  predictions. A model that is confident in its wrong predictions is
  more likely to make mistakes in the future. Log-loss penalizes models
  for being confident in their wrong predictions, which encourages them
  to be more cautious.  
- It is **robust to imbalanced data sets**. When a data set is
  imbalanced (which is the case here), one class may be much more common
  than the other class. This can lead to models that are biased towards
  predicting the majority class. Accuracy is not robust to imbalanced
  data sets, as it can be high even if the model is simply predicting
  the majority class all the time. Log-loss is robust to imbalanced data
  sets, as it penalizes models for being confident in their wrong
  predictions, even if they are predicting the majority class.  
- It is more **sensitive to small changes in model performance**.
  Accuracy is a step function, meaning that it only changes when a model
  makes a mistake. Log-loss is a continuous function, meaning that it
  changes even when a model’s predictions are slightly off. This makes
  log-loss a more sensitive metric for evaluating model performance, and
  it can be used to identify areas where a model can be improved.

In general, log-loss is a better metric than accuracy for evaluating the
performance of models. Since the data set we are working on for this
project is imbalanced, it is ideal to use Log-loss as the model metric
when compared to accuracy.

### Logistic Regression

Generalized linear models (GLM) are used to model non-normal
distributions. Logistic regression is a very common generalized linear
model that is used to predict the probability of a binary outcome
(success or fail). This type of model is used for both classification
and regression. Since the outcome is a probability, the response
variable is constrained to values between 0 and 1.

Here the probability of success is modeled using a function that does
not have a closed form solution. Hence, maximum likelihood is often used
to fit the parameters. In a logistic model, a log transformation of this
function is performed, usually called log-odds or the *logit function*.
The logit function is the *link* that linearly associates mean of
response variable to the parameters in model.

For the current data set, our variable of interest- Diabetes_binary, is
a binomial random variable i.e. it has a value of either 0 or 1.
Logistic regression would be an ideal model to apply here.

We applied three candidate logistic models here: forward selection
model, backward selection model, and a basic logistic regression model
(glm).

#### Model 1- Forward Selection

Here we used the *forward selection method* to construct a regression
model. We trained the model with all variables from the training data
set. Only main effects were considered here. Interaction terms and
higher polynomial terms were not considered. The model was trained with
the following parameters:  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
# Model 1: Forward Selection
model_1_forward <- train( Diabetes_binary ~ .,
                          data = diabetes_train,
                          method = "glmStepAIC",
                          family = "binomial",
                          direction = "forward",
                          metric = "logLoss",
                          trace = FALSE,
                          preProcess = c("center", "scale"),
                          trControl = 
                            trainControl(method = "cv", number = 5, 
                                         summaryFunction = mnLogLoss, 
                                         classProbs = TRUE))

model_1_forward
```

    ## Generalized Linear Model with Stepwise Feature Selection 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## Pre-processing: centered (35), scaled (35) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2360, 2362, 2362, 2362, 2362 
    ## Resampling results:
    ## 
    ##   logLoss 
    ##   0.522748

The Logloss value of Forward selection model is 0.522748.

#### Model 2- Backward Selection Method

Here we used the *backward selection method* to construct a regression
model. We trained the model with all variables from the training data
set. Only main effects were considered here. Interaction terms and
higher polynomial terms were not considered. The model was trained with
the following parameters:  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
# Model 2: Backward Selection
model_2_backward <- train( Diabetes_binary ~ .,
                           data = diabetes_train,
                           method = "glmStepAIC",
                           family = "binomial",
                           direction = "backward",
                           metric = "logLoss",
                           trace = FALSE,
                           preProcess = c("center", "scale"),
                           trControl = 
                             trainControl(method = "cv", number = 5,
                                          summaryFunction = mnLogLoss, 
                                          classProbs = TRUE))

model_2_backward
```

    ## Generalized Linear Model with Stepwise Feature Selection 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## Pre-processing: centered (35), scaled (35) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2360, 2362, 2362, 2362 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.5214899

The Logloss value of Backward selection model is 0.5214899.

#### Model 3

Here we fit a basic logistic regression model. We trained the model with
all variables from the training data set. Only main effects were
considered here. Interaction terms and higher polynomial terms were not
considered. The model was trained with the following parameters:  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
# Model 3: Generalized Linear Model
model_3_glm <- train( Diabetes_binary ~ ., 
                       data = diabetes_train,
                       method = "glm",
                       metric = "logLoss",
                       trace = FALSE,
                       preProcess = c("center", "scale"),
                       trControl = 
                         trainControl(method = "cv", number = 5,
                                      summaryFunction = mnLogLoss,
                                      classProbs = TRUE))

model_3_glm
```

    ## Generalized Linear Model 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## Pre-processing: centered (35), scaled (35) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2361, 2361, 2362 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.5260225

The Logloss value of Backward selection model is 0.5260225.

#### Comparison of Models

The logloss values for all 3 models is tabulated below.

``` r
result1 <- data.frame(Model_1 = model_1_forward$results$logLoss, 
            Model_2 = model_2_backward$results$logLoss,
            Model_3 = model_3_glm$results$logLoss)

result1[1,]
```

    ##    Model_1   Model_2   Model_3
    ## 1 0.522748 0.5214899 0.5260225

The lowest logloss value is for Model_2 with a value of 0.5214899. We
choose **Model_2** as the best model.

#### Best Model Results

Here we created a R function, *best_result*, that takes the 3 model fits
from above as input, then based on the logloss value outputs the best
model fit. This function was created to automate the process of choosing
the best across all reports.

``` r
best_result <- function(model1, model2, model3){
  name <- names(result1)[apply(result1, MARGIN = 1, FUN = which.min)]
  if(name == "Model_1"){
    fit <- model1
  }
  if(name == "Model_2") {
    fit <- model2
  }
  if(name == "Model_3") {
    fit <- model3
  }
  return(fit)
}

best_model1 <- best_result(model_1_forward, model_2_backward, model_3_glm)
```

### LASSO Logistic Regression Model

LASSO or Least Absolute Shrinkage and Selection Operator, is a
regression analysis model that *performs both variable selection and
regularization* to improve prediction accuracy. It is a **penalized**
regression approach that estimates the regression coefficients by
minimizing the sum of squared residuals and the sum of the absolute
values of the regression coefficients multiplied with a positive
constant, $\lambda$. Mathematically it can be expressed as:

$$L +\lambda \sum_{i = 1}^{n} |\beta_i|$$ where L is the sum of squared
residuals, and $\lambda \sum_{i = 1}^{n} |\beta_i|$ is the penalty.

- when $\lambda$ = 0, all variables are included  
- when $\lambda$ = $\infty$, no variables are chosen  
- usually cross-validation is used to choose the value of $\lambda$ such
  that some coefficients will shrink to 0

A LASSO Model uses a **L1 Regularization technique**. Regularization
refers to techniques that are used to calibrate models in order to
minimize modified sum of squared errors (above equation) and prevent
over fitting or under fitting data.

In a linear model like basic logistic regression (essentially $\lambda$
= 0 or no penalty), they tend to have some variance, that is the model
does not generalize well to data other than training set. In the case of
a LASSO model, regularization significantly reduces variance of the
model without an increase in bias, a balance of bias-variance trade off.
This is achieved with the help of the tuning parameter, $\lambda$. As
$\lambda$ increases, it reduces the value of coefficients and thus
reducing the variance of the model, avoiding an over fit. But after
certain increase in $\lambda$, we loose important features of the data
and this will increase the bias resulting in under fitting. Hence, an
optimal value of $\lambda$ is chosen using cross-validation.

Here a LASSO logistic regression model was fit with the following
parameters:  
+ $\alpha$ = 1  
+ $\lambda$ : a sequence of numbers from 0 to 1 with an increment of
0.01  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
lasso_fit <- train(Diabetes_binary  ~ ., 
              data = diabetes_train,
              method = "glmnet",
              metric = "logLoss",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 5, 
                                       summaryFunction = mnLogLoss, classProbs = TRUE),
              tuneGrid = expand.grid(alpha = 1, lambda=seq(0,1 , by = 0.01)))

lasso_fit
```

    ## glmnet 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## Pre-processing: centered (35), scaled (35) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2362, 2361, 2361 
    ## Resampling results across tuning parameters:
    ## 
    ##   lambda  logLoss  
    ##   0.00    0.5208757
    ##   0.01    0.5227272
    ##   0.02    0.5310202
    ##   0.03    0.5395919
    ##   0.04    0.5480577
    ##   0.05    0.5562392
    ##   0.06    0.5638470
    ##   0.07    0.5724841
    ##   0.08    0.5811661
    ##   0.09    0.5882041
    ##   0.10    0.5943305
    ##   0.11    0.5999149
    ##   0.12    0.6036376
    ##   0.13    0.6036376
    ##   0.14    0.6036376
    ##   0.15    0.6036376
    ##   0.16    0.6036376
    ##   0.17    0.6036376
    ##   0.18    0.6036376
    ##   0.19    0.6036376
    ##   0.20    0.6036376
    ##   0.21    0.6036376
    ##   0.22    0.6036376
    ##   0.23    0.6036376
    ##   0.24    0.6036376
    ##   0.25    0.6036376
    ##   0.26    0.6036376
    ##   0.27    0.6036376
    ##   0.28    0.6036376
    ##   0.29    0.6036376
    ##   0.30    0.6036376
    ##   0.31    0.6036376
    ##   0.32    0.6036376
    ##   0.33    0.6036376
    ##   0.34    0.6036376
    ##   0.35    0.6036376
    ##   0.36    0.6036376
    ##   0.37    0.6036376
    ##   0.38    0.6036376
    ##   0.39    0.6036376
    ##   0.40    0.6036376
    ##   0.41    0.6036376
    ##   0.42    0.6036376
    ##   0.43    0.6036376
    ##   0.44    0.6036376
    ##   0.45    0.6036376
    ##   0.46    0.6036376
    ##   0.47    0.6036376
    ##   0.48    0.6036376
    ##   0.49    0.6036376
    ##   0.50    0.6036376
    ##   0.51    0.6036376
    ##   0.52    0.6036376
    ##   0.53    0.6036376
    ##   0.54    0.6036376
    ##   0.55    0.6036376
    ##   0.56    0.6036376
    ##   0.57    0.6036376
    ##   0.58    0.6036376
    ##   0.59    0.6036376
    ##   0.60    0.6036376
    ##   0.61    0.6036376
    ##   0.62    0.6036376
    ##   0.63    0.6036376
    ##   0.64    0.6036376
    ##   0.65    0.6036376
    ##   0.66    0.6036376
    ##   0.67    0.6036376
    ##   0.68    0.6036376
    ##   0.69    0.6036376
    ##   0.70    0.6036376
    ##   0.71    0.6036376
    ##   0.72    0.6036376
    ##   0.73    0.6036376
    ##   0.74    0.6036376
    ##   0.75    0.6036376
    ##   0.76    0.6036376
    ##   0.77    0.6036376
    ##   0.78    0.6036376
    ##   0.79    0.6036376
    ##   0.80    0.6036376
    ##   0.81    0.6036376
    ##   0.82    0.6036376
    ##   0.83    0.6036376
    ##   0.84    0.6036376
    ##   0.85    0.6036376
    ##   0.86    0.6036376
    ##   0.87    0.6036376
    ##   0.88    0.6036376
    ##   0.89    0.6036376
    ##   0.90    0.6036376
    ##   0.91    0.6036376
    ##   0.92    0.6036376
    ##   0.93    0.6036376
    ##   0.94    0.6036376
    ##   0.95    0.6036376
    ##   0.96    0.6036376
    ##   0.97    0.6036376
    ##   0.98    0.6036376
    ##   0.99    0.6036376
    ##   1.00    0.6036376
    ## 
    ## Tuning parameter 'alpha' was held constant at a value of 1
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final values used for the model were alpha = 1 and lambda = 0.

The plot below shows the model performance with different $\lambda$
values.

``` r
plot(lasso_fit)
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

The best LASSO model has an optimal $\lambda$ value of 0 with a logLoss
of 0.5208757.

### Classification Tree Model

Tree based method is a *non-linear* regression model used for
prediction. In a tree based method, the predictor space is *split into
regions*, and each region has a different prediction. For a given
region, the most prevalent class is used as the prediction in a
classification problem. While the mean of observations in the region is
used as prediction in the case of a continuous variable.

Tree based methods use a greedy algorithm called **recursive binary
splitting** to pick the splits. For every possible value of each
predictor, they find the residual sum of squares (RSS) or Gini
index/Deviance (for classification) and try to minimize it. The split it
made at the location where RSS is split. Then the algorithm repeats the
process till we have a large tree with several nodes. This tree is then
pruned back using **cost-complexity pruning** so that we do not over fit
the model to training data. The pruning process increases the bias but
decreases the variance, finding a balance of bias-variance trade off.
Cross-validation can help choosing the optimum number of nodes in a
tree.

Tree based methods are flexible, very intuitive and easy to read. They
also do not need interactions terms in the model. This makes them an
ideal candidate to use here.

Here a classification tree model was fit with the following
parameters:  
+ complexity parameter : a sequence of numbers from 0 to 0.1 with an
increment of 0.001  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
classification_tree <- train(Diabetes_binary ~ ., 
                 data = diabetes_train,
                 method = "rpart",
                 metric = "logLoss",
                 trControl = trainControl(method = "cv", number = 5, 
                                          summaryFunction = mnLogLoss, classProbs = TRUE),
                 tuneGrid = expand.grid(cp = seq(from = 0, to = 0.1, by = 0.001)))

classification_tree
```

    ## CART 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2360, 2362, 2362, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     logLoss  
    ##   0.000  0.7209417
    ##   0.001  0.6909735
    ##   0.002  0.6765799
    ##   0.003  0.5770662
    ##   0.004  0.5743653
    ##   0.005  0.5695223
    ##   0.006  0.5697872
    ##   0.007  0.5697937
    ##   0.008  0.5675831
    ##   0.009  0.5673349
    ##   0.010  0.5673105
    ##   0.011  0.5659342
    ##   0.012  0.5681261
    ##   0.013  0.5681261
    ##   0.014  0.5681261
    ##   0.015  0.5675984
    ##   0.016  0.5672033
    ##   0.017  0.5672033
    ##   0.018  0.5672033
    ##   0.019  0.5672033
    ##   0.020  0.5780441
    ##   0.021  0.5780441
    ##   0.022  0.5780441
    ##   0.023  0.5859450
    ##   0.024  0.5859450
    ##   0.025  0.5859450
    ##   0.026  0.5859450
    ##   0.027  0.5859450
    ##   0.028  0.5859450
    ##   0.029  0.5859450
    ##   0.030  0.5949481
    ##   0.031  0.5949481
    ##   0.032  0.6015429
    ##   0.033  0.6015429
    ##   0.034  0.6015429
    ##   0.035  0.6036370
    ##   0.036  0.6036370
    ##   0.037  0.6036370
    ##   0.038  0.6036370
    ##   0.039  0.6036370
    ##   0.040  0.6036370
    ##   0.041  0.6036370
    ##   0.042  0.6036370
    ##   0.043  0.6036370
    ##   0.044  0.6036370
    ##   0.045  0.6036370
    ##   0.046  0.6036370
    ##   0.047  0.6036370
    ##   0.048  0.6036370
    ##   0.049  0.6036370
    ##   0.050  0.6036370
    ##   0.051  0.6036370
    ##   0.052  0.6036370
    ##   0.053  0.6036370
    ##   0.054  0.6036370
    ##   0.055  0.6036370
    ##   0.056  0.6036370
    ##   0.057  0.6036370
    ##   0.058  0.6036370
    ##   0.059  0.6036370
    ##   0.060  0.6036370
    ##   0.061  0.6036370
    ##   0.062  0.6036370
    ##   0.063  0.6036370
    ##   0.064  0.6036370
    ##   0.065  0.6036370
    ##   0.066  0.6036370
    ##   0.067  0.6036370
    ##   0.068  0.6036370
    ##   0.069  0.6036370
    ##   0.070  0.6036370
    ##   0.071  0.6036370
    ##   0.072  0.6036370
    ##   0.073  0.6036370
    ##   0.074  0.6036370
    ##   0.075  0.6036370
    ##   0.076  0.6036370
    ##   0.077  0.6036370
    ##   0.078  0.6036370
    ##   0.079  0.6036370
    ##   0.080  0.6036370
    ##   0.081  0.6036370
    ##   0.082  0.6036370
    ##   0.083  0.6036370
    ##   0.084  0.6036370
    ##   0.085  0.6036370
    ##   0.086  0.6036370
    ##   0.087  0.6036370
    ##   0.088  0.6036370
    ##   0.089  0.6036370
    ##   0.090  0.6036370
    ##   0.091  0.6036370
    ##   0.092  0.6036370
    ##   0.093  0.6036370
    ##   0.094  0.6036370
    ##   0.095  0.6036370
    ##   0.096  0.6036370
    ##   0.097  0.6036370
    ##   0.098  0.6036370
    ##   0.099  0.6036370
    ##   0.100  0.6036370
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.011.

The plot below shows the model performance with different complexity
parameter values.

``` r
 plot(classification_tree)
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

The best classification tree model has an optimal complexity parameter
value of 0.011 with a logLoss of 0.5659342.

### Random Forest Model

Random forest is a supervised non-linear machine learning algorithm that
uses a collection of decision trees from bootstrap samples and averages
the results to make predictions. Rather than using all the predictors
for the trees, it takes a randomly selected subset of variables for each
bootstrap sample/ tree fit. By using only a subset of randomly chosen
predictors, the random forest model reduces the number of correlated
trees and hence an overall reduction in variance occurs.

Random forests differ from decision trees in two main ways:

- Random forests use multiple decision trees. This is called ensemble
  learning. By averaging the predictions of multiple decision trees,
  random forests can reduce the risk of over fitting and improve the
  overall accuracy of the model.  
- Random forests use a random subset of features to build each decision
  tree. This is called feature bagging. Feature bagging helps to reduce
  the correlation between the decision trees, which further improves the
  accuracy of the model.

Here the random forest method was used to fit the data with the
following parameters:  
+ tuning parameter, mtry: 1 through 5 with an increment of 1. This
number was arrived by dividing number of predictors by 3  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
# Random Forest Model
rf_model <- train( Diabetes_binary ~ .,
                   data = diabetes_train,
                   method = "rf",
                   metric = "logLoss",
                   trControl = 
                     trainControl(method = "cv", number = 5,
                                  summaryFunction = mnLogLoss, classProbs = TRUE),
                   tuneGrid = data.frame(mtry = seq(1:5)))

rf_model
```

    ## Random Forest 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2361, 2361, 2362, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  logLoss  
    ##   1     0.8337663
    ##   2     0.5593782
    ##   3     0.5293308
    ##   4     0.5274073
    ##   5     0.5278843
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 4.

The plot below shows the model performance with different number of
randomly selected predictors.

``` r
plot(rf_model)
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

The best random forest model has 4 randomly selected predictors with a
logLoss of 0.5274073.

### Partial Least Squares Model

Partial Least Squares (PLS) is a statistical method that is often used
in regression and classification analysis. PLS regression is commonly
employed when there are high collinearity among the predictor variables,
and it aims to find the directions (latent variables or components) that
explain both the variance in the predictor variables and the variance in
the response variable.

Partial Least Squares Discriminant Analysis (PLS-DA) is an extension of
PLS that is specifically used for classification purposes. PLS-DA
combines elements of principal component analysis (PCA) and canonical
correlation analysis to find the linear combinations of the original
variables (features) that best discriminate between different classes in
the response variable.

In the context of classification, the goal of PLS-DA is to find a set of
latent variables (components) that maximize the separation between
different classes while also explaining the variance in the predictor
variables. These latent variables are then used to build a predictive
model for classifying new observations into predefined classes.

PLS and PLS-DA are commonly used in fields such as chemometrics,
biology, and other areas where there are complex relationships between
variables and a need for effective classification or regression models.
They are particularly useful when dealing with high-dimensional data or
when there are multicollinearity issues among the predictor variables.

Here the pls method was used to fit the data with the following
parameters:  
+ ncomp set to values 0 through 15  
+ cross-validation with 5 folds  
+ log-loss as metric to evaluate model

``` r
# Partial Least Squares Model
pls_model <- train( Diabetes_binary ~ .,
                    data = diabetes_train,
                    method = "pls",
                    metric = "logLoss",
                    trControl = trainControl(method = "cv", number = 5,
                                             preProcOptions = c("center","scale"),
                                             summaryFunction = mnLogLoss,
                                             classProbs = TRUE),
                    tuneGrid = data.frame(ncomp = seq(1:15)))

pls_model
```

    ## Partial Least Squares 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2360, 2362, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   ncomp  logLoss  
    ##    1     0.6149725
    ##    2     0.6108623
    ##    3     0.6098588
    ##    4     0.5826268
    ##    5     0.5796004
    ##    6     0.5782880
    ##    7     0.5775187
    ##    8     0.5770192
    ##    9     0.5770379
    ##   10     0.5770693
    ##   11     0.5769522
    ##   12     0.5768876
    ##   13     0.5769155
    ##   14     0.5768981
    ##   15     0.5768902
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was ncomp = 12.

The plot below shows the model performance with different ncomp values.

``` r
plot(pls_model)
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

The best pls model has 12 number of components in the model with a
logLoss of 0.5768876.

### Naive Bayes Model

Naive Bayes model, also known as a probabilistic classifier, is based on
Bayes Theorem with an independence assumption among predictors. Bayes
theorem in simple terms involves making an initial assumption of
probability (prior probability), and then conditionally updating this
probability based on new data or evidence. This prior probability is
sequentially updated when new evidence emerges.

Naive Bayes classifier makes two key assumptions: all predictors are
conditionally independent or unrelated, and all predictors contribute
equally to the outcome or response. Although these assumptions do not
hold well in real-world, they are computationally fast and have shown to
have good accuracy in predictions. We wanted to see how naive bayes
model works on our data.

Here the naive bayes method was used to fit the data with the following
parameters:  
+ fL values 0 and 0.5  
+ useKernel = TRUE  
+ adjust parameter set to vector (1.0, 2.0, 4.0) + cross-validation with
5 folds  
+ log-loss as metric to evaluate model

``` r
naive_bayes_model <- train( Diabetes_binary ~ .,
                    data = diabetes_train,
                    method = "nb",
                    metric = "logLoss",
                    trControl = trainControl(method = "cv", number = 5,
                                             summaryFunction = mnLogLoss,
                                             classProbs = TRUE),
                    tuneGrid = expand.grid(fL = c(0,0.5), usekernel = TRUE, adjust = c(1.0, 2.0, 4.0) ))
  
naive_bayes_model
```

    ## Naive Bayes 
    ## 
    ## 2952 samples
    ##   15 predictor
    ##    2 classes: 'Diabetes', 'No_Diabetes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2362, 2360, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   fL   adjust  logLoss  
    ##   0.0  1       0.7389652
    ##   0.0  2       0.7348858
    ##   0.0  4       0.6822610
    ##   0.5  1       0.7389652
    ##   0.5  2       0.7348858
    ##   0.5  4       0.6822610
    ## 
    ## Tuning parameter 'usekernel' was held constant at a value of TRUE
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust = 4.

The plot below shows the model performance.

``` r
plot(naive_bayes_model)
```

![](Education%20Group%202_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

The best naive bayes model has parameters: fl- 0, and adjust- 4 with a
logLoss of 0.682261.

## Final Model Selection

For this section we have compiled the model output results of the 6 best
models so far. We then applied these model fits on test data set to see
their performance. The models considered here included:  
+ Model A: Best of Logistic Regression Model  
+ Model B: LASSO Model  
+ Model C: Classification Tree  
+ Model D: Random Forest  
+ Model E: Partial Least Squares  
+ Model F: Naive Bayes

### Output from Model Comparison

Model output results are summarized below:

``` r
# Best model from the first three models (forward, backward, and glm)
model_name <- names(result1)[apply(result1, MARGIN = 1, FUN = which.min)]
value <- min(result1)

result2 <- data.frame(model_name = value, 
            LASSO = min(lasso_fit$results$logLoss),
            Classification_Tree = min( classification_tree$results$logLoss),
            Random_Forest = min( rf_model$results$logLoss),
            Partial_Least_Squares = min( pls_model$results$logLoss),
            Naive_Bayes = min(naive_bayes_model$results$logLoss))

colnames(result2)[1] <- model_name
result2[1,]
```

    ##     Model_2     LASSO Classification_Tree Random_Forest Partial_Least_Squares
    ## 1 0.5214899 0.5208757           0.5659342     0.5274073             0.5768876
    ##   Naive_Bayes
    ## 1    0.682261

Based on the results from model fits, the best model is **LASSO** with a
logloss value of **`{r} min(result2)`**. Note, these are the results
model fits.

### Application of Model on Test Data

Here we fit the 6 best models on test data set and compared the results
using logloss values as the metric. The results of all model predictions
on test data were stored in the R Object, *result3*

``` r
# Log loss calculation for Model 1 (best of the first 3 Models)
model_a_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(best_model1, newdata = diabetes_test),
                                  predict(best_model1, diabetes_test,type="prob"))

model_a_pred_metric <- mnLogLoss(model_a_pred, lev = levels(model_a_pred$obs))


# Log loss calculation for LASSO
model_b_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(lasso_fit, newdata = diabetes_test),
                                  predict(lasso_fit, diabetes_test,type="prob"))

model_b_pred_metric <- mnLogLoss(model_b_pred, lev = levels(model_b_pred$obs))

# Log loss calculation for Classification Tree- 
model_c_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(classification_tree, newdata = diabetes_test),
                                  predict(classification_tree, diabetes_test,type="prob"))

model_c_pred_metric <- mnLogLoss(model_c_pred, lev = levels(model_c_pred$obs))

# Log loss calculation for Random Forest- 
model_d_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(rf_model, newdata = diabetes_test),
                                  predict(rf_model, diabetes_test,type="prob"))

model_d_pred_metric <- mnLogLoss(model_d_pred, lev = levels(model_d_pred$obs))

# Log loss calculation for Partial Squares- 
model_e_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(pls_model, newdata = diabetes_test),
                                  predict(pls_model, diabetes_test,type="prob"))

model_e_pred_metric <- mnLogLoss(model_e_pred, lev = levels(model_e_pred$obs))

# Log loss calculation for Naive Bayes- 
model_f_pred <- data.frame(obs=diabetes_test$Diabetes_binary, 
                                  pred=predict(naive_bayes_model, newdata = diabetes_test),
                                  predict(naive_bayes_model, diabetes_test,type="prob"))

model_f_pred_metric <- mnLogLoss(model_f_pred, lev = levels(model_f_pred$obs))


# The results are compiled into one data frame here
result3 <- data.frame(Model_A = model_a_pred_metric,
                      LASSO = model_b_pred_metric,
                      Classification_tree = model_c_pred_metric,
                      Random_Forest = model_d_pred_metric,
                      Partial_Squares = model_e_pred_metric,
                      Naive_Bayes = model_f_pred_metric)
colnames(result3)[1] <- model_name

result3[1,]
```

    ##           Model_2     LASSO Classification_tree Random_Forest Partial_Squares
    ## logLoss 0.5231097 0.5227593           0.5624469      0.541524       0.5813912
    ##         Naive_Bayes
    ## logLoss   0.7125617

### Result

Based on the model predictions on test data set, the best model is
**LASSO** with a logloss value of **0.5227593**.
