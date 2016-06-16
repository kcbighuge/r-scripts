# xgboost intro
# http://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
# also: http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html

require(xgboost)
require(DiagrammeR)

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test

str(train)
str(test)

model <- xgboost(data=train$data, label=train$label, 
                 nrounds=2, objective='binary:logistic')
preds <- predict(model, test$data)
str(preds)

cv.res <- xgb.cv(data=train$data, label=train$label,
                 nfold=5, nrounds=2, objective='binary:logistic')

### CUSTOMIZED OBJECTIVE ###
loglossobj <- function(preds, dtrain) {
  # extract labels from training data
  labels <- getinfo(dtrain, 'label')
  
  # compute 1st and 2nd gradient as grad, hess
  preds <- 1 / (1+exp(-preds))
  grad <- preds - labels
  hess <- preds * (1-preds)
  
  # return result as list
  return(list(grad=grad, hess=hess))
}

# train the model
model <- xgboost(data=train$data, label=train$label,
                 nrounds=2, objective=loglossobj, eval_metric='error')

### EARLY STOPPING ###
bst <- xgb.cv(data=train$data, label=train$label,
              nfold=5, nrounds = 32, objective='binary:logistic',
              early.stop.round=8, maximize = FALSE)

### CONTINUE TRAINING ###
# users can continue the training on the previous model,
# thus the 2nd step will cost you the time for the addl iterations only
# `xgb.DMatrix` object contains the features, target,
#   and other side informations, e.g. weights, missing values.
dtrain <- xgb.DMatrix(train$data, label=train$label)
model <- xgboost(data = dtrain, nrounds=2, objective='binary:logistic')

# predict the current training data
# parameter 'outputmargin' indicates that 
# we don’t need a logistic transformation of the result
pred_train <- predict(model, dtrain, outputmargin=TRUE)

# put the previous prediction result as an addl info to object `dtrain`,
# so that the training algorithm knows where to start.
setinfo(dtrain, "base_margin", pred_train)

# now observe results change
model <- xgboost(data=dtrain, nrounds=2, objective='binary:logistic')


### HANDLE MISSING VALUES ###
# When using a feature with missing values to do splitting,
# xgboost will assign a direction to the missing values
# instead of a numerical value. Specifically, xgboost guides all
# the data points with missing values to the left and right respectively
# then choose the direction with a higher gain with regard to the objective.

# generate data
dat <- matrix(rnorm(128), 64, 2)
label <- sample(0:1, nrow(dat), replace=TRUE)
for (i in 1:nrow(dat)) {
  ind <- sample(2, 1)
  dat[i, ind] <- NA
}
str(dat)

# set the parameter missing to mark the missing value label
# specify missing value marker "NA"
model <- xgboost(data=dat, label=label, missing=NA,
                 nrounds=2, objective='binary:logistic')


########################
### MODEL INSPECTION ###
########################

# model usually contains multiple trees
bst <- xgboost(data=train$data, label=train$label,
               max.depth=2, eta=1, nthread=2, nround=2, objective='binary:logistic')
xgb.plot.tree(feature_names = agaricus.train$data@Dimnames[[2]], model=bst)

# what if we have more trees?
bst <- xgboost(data=train$data, label=train$label,
               max.depth=2, eta=1, nthread=2, nround=8, objective='binary:logistic')
xgb.plot.tree(feature_names=agaricus.train$data@Dimnames[[2]], model=bst)

### MULTIPLE-IN-ONE PLOT ###
# 1. Almost all the trees in an ensemble model have the same shape. 
#   If the maximum depth is determined, this holds for all the binary trees.
# 2. On each node there would be more than one feature that have appeared 
#   on this position. But we can describe it by the frequency of each feature 
#   thus make a frequency table.
bst <- xgboost(data=train$data, label = train$label, max.depth=15, eta=1, 
               nthread=2, nround=32, objective='binary:logistic',
               min_child_weight=50)
# mouseover on the nodes to get information of the path
xgb.plot.multi.trees(model = bst, 
                     feature_names = agaricus.train$data@Dimnames[[2]],
                     features.keep = 3)

### FEATURE IMPORTANCE ###
# calculate the gain on each node, the contribution from the selected feature
# look into all the trees, and sum up all the contribution for each feature
# can also do a clustering on features before we make the plot
bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
               eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
imp_matrix <- xgb.importance(agaricus.train$data@Dimnames[[2]], model=bst)
xgb.plot.importance(imp_matrix)

### DEEPNESS ###
# get two plots summarizing the distribution of leaves
# according to the change of depth in the tree
bst <- xgboost(data = train$data, label = train$label, max.depth = 4,
               eta = 1, nthread = 2, nround = 30, objective = "binary:logistic",
               min_child_weight = 50)
# upper plot: the number of leaves per level of deepness.
# lower plot: noramlized weighted cover per leaf (weighted sum of instances)
xgb.plot.deepness(model=bst)



