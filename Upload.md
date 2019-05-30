Practical Machine Learning - Prediction Assignment Writeup
==========================================================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har" class="uri">http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har</a>
(see the section on the Weight Lifting Exercise Dataset).

Data
----

The training data for this project are available here:
<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a>

The test data are available here:
<a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" class="uri">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv</a>

Data analysis
=============

Loading the data
----------------

``` r
train<- read.csv("pml-training.csv")
test<- read.csv("pml-testing.csv")
cat("Dimensions of training dataset")
```

    ## Dimensions of training dataset

``` r
dim(train)
```

    ## [1] 19622   160

``` r
cat("Dimensions of testing dataset")
```

    ## Dimensions of testing dataset

``` r
dim(test)
```

    ## [1]  20 160

Variables with majority of missing values need to be excluded, as it can
be anticipated that new data would also be unknown and therefore useless
for prediction. Therefore all variables with missing values are
excluded, then ID type variables and non numerics are excluded

``` r
missing_train = colSums(is.na(train))
train <- train[, missing_train == 0] 
test <- test[, missing_train == 0] 
train_outcome <- train$classe
test_outcome <- test$classe
train <- train[, 6:dim(train)[2]]
test <- test[, 6:dim(test)[2]]
train <- train[, sapply(train, is.numeric)]
train$outcome <- train_outcome
test <- test[, sapply(test, is.numeric)]
test$outcome <- test_outcome
```

Data spliting for CV
--------------------

To train models cross-validation is used using a 70/30 split

``` r
set.seed(42) 
inTrain <- createDataPartition(train$outcome, p=0.70, list=F)
train_data <- train[inTrain, ]
test_data <- train[-inTrain, ]
```

Model building
--------------

One of the most powerfull methods is the random forests. Therefore it
will be trainied on the training dataset.

``` r
set.seed(42) 
setting <- trainControl(method="cv", 5)
RandomForest <- train(outcome ~ ., data=train_data, method="rf", trControl=setting, ntree=100)
RandomForest
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10991, 10991, 10990, 10988, 10988 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9922838  0.9902385
    ##   27    0.9974521  0.9967772
    ##   53    0.9940310  0.9924495
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

We use this trained model on the test set to see its accuracy

``` r
predicted <- predict(RandomForest, test_data)
confusionMatrix(test_data$outcome, predicted)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    0 1137    2    0    0
    ##          C    0    3 1023    0    0
    ##          D    0    0    4  959    1
    ##          E    0    0    0    5 1077
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9975          
    ##                  95% CI : (0.9958, 0.9986)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9968          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9974   0.9942   0.9948   0.9991
    ## Specificity            1.0000   0.9996   0.9994   0.9990   0.9990
    ## Pos Pred Value         1.0000   0.9982   0.9971   0.9948   0.9954
    ## Neg Pred Value         1.0000   0.9994   0.9988   0.9990   0.9998
    ## Prevalence             0.2845   0.1937   0.1749   0.1638   0.1832
    ## Detection Rate         0.2845   0.1932   0.1738   0.1630   0.1830
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      1.0000   0.9985   0.9968   0.9969   0.9990

``` r
accuracy <- postResample(predicted, test_data$outcome)
error<-1 - as.numeric(confusionMatrix(test_data$outcome, predicted)$overall[1])
```

The accuracy is: 0.9968 while the out of test error is: 0.032

Application to the test data
============================

``` r
predict(RandomForest, test)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
