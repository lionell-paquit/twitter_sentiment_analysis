install.packages("tm") # if tm package is not yet installed
install.packages("e1071") # if e1071 package is not yet installed
install.packages("SnowBallC") # if SnowballC package is not yet installed
install.packages("RTextTools") # if RTextTools package is not yet installed

libs <- c("tm", "e1071", "SnowballC", "RTextTools")
lapply(libs, require, character.only = TRUE) #apply required library

options(stringAsFactors = FALSE)

setwd("D:/DATA/Graduate/Tweet Analysis Demo/data")

tmpTrainTweet <- read.csv("training.csv", header = FALSE,
                          stringsAsFactors = FALSE)   # read data from training.csv
testTweet <- read.csv("test.csv", header = FALSE,
                      stringsAsFactors=FALSE)         # read data from test.csv

tmpPosTweet <- tmpTrainTweet[tmpTrainTweet[, 1] == 4, c(1, 6)]
tmpNegTweet <- tmpTrainTweet[tmpTrainTweet[, 1] == 0, c(1, 6)]

trainingSize <- 4000 # set training size
set.seed(12345) # seed is any integer used for random sampling

posTweet <- tmpPosTweet[sample(nrow(tmpPosTweet),
                               size = trainingSize/2, replace = FALSE), ] # get random sample from positive tweets
negTweet <- tmpNegTweet[sample(nrow(tmpNegTweet),
                               size = trainingSize/2, replace = FALSE), ] # get random sample from negative tweets

trainTweet <- rbind(posTweet, negTweet) # combine positive and negative tweets

testTweet <- testTweet[testTweet[, 1] != 2, ]
tweetAll <- rbind(trainTweet, testTweet[ , c(1, 6)])

colnames(tweetAll) <- c("polarity", "tweet")

tweetAll$polarity <- replace(tweetAll$polarity, tweetAll$polarity==4, "positive")
tweetAll$polarity <- replace(tweetAll$polarity, tweetAll$polarity==0, "negative")

tweetAll$tweet <- gsub("@\\w+", "", tweetAll$tweet) # remove @username
tweetAll$tweet <- gsub("http\\S+\\s*", "", tweetAll$tweet) #  remove URL

# create a document-term matrix and pre-processed data
matAll <- create_matrix(tweetAll$tweet, language="english", toLower = TRUE,
                       removeStopwords = TRUE, removeNumbers = TRUE,
                       stripWhitespace = TRUE, removePunctuation = TRUE,
                       removeSparseTerms = 0.999, 
                       stemWords = TRUE, tm::weightTfIdf)

saveRDS(tweetAll, "tweetAll.rds") # save tweet object
saveRDS(matAll, "matAll.rds")     # save document-term matrix object

rm(list = ls()) # remove unnecessary variables to free up RAM

tweetAll <- readRDS("tweetAll.rds") # read and load tweet object
matAll <- readRDS("matAll.rds") # read and load document-term matrix object

trainingSize <- 4000 # training set size

# Naive Bayes training and classification
matAll <- as.matrix(matAll)

ptm <- proc.time()
nbModel <- naiveBayes(matAll[1:trainingSize, ], as.factor(tweetAll$polarity[1:trainingSize]))
NBprocTime <- proc.time() - ptm

nbClassify <- predict(nbModel, matAll[(trainingSize+1):nrow(tweetAll), ])

# MAXENT, SVM, RF, TREE
matAll <- readRDS("matAll.rds")

container <- create_container(matAll, as.numeric(as.factor(tweetAll$polarity)),
                             trainSize=1:trainingSize, testSize=(trainingSize+1):nrow(tweetAll),
                             virgin=FALSE)

#train the algorithms
ptm <- proc.time()
maxentModel <- train_model(container, "MAXENT")
maxentprocTime <- proc.time() - ptm

ptm <- proc.time()
svmModel <- train_model(container, "SVM")
svmprocTime <- proc.time() - ptm

ptm <- proc.time()
treeModel <- train_model(container, "TREE")
treeprocTime <- proc.time() - ptm

ptm <- proc.time()
rfModel10 <- train_model(container, "RF", ntree = 10)
rfprocTime <- proc.time() - ptm

ptm <- proc.time()
rfModel20 <- train_model(container, "RF", ntree = 20)
rfprocTime <- proc.time() - ptm

ptm <- proc.time()
rfModel <- train_model(container, "RF")
rfprocTime <- proc.time() - ptm

#classify the testing data using the trained models

maxentClassify <- classify_model(container, maxentModel)

svmClassify <- classify_model(container, svmModel)

treeClassify <- classify_model(container, treeModel)

rfClassify <- classify_model(container, rfModel)

rfClassify10 <- classify_model(container, rfModel10)

rfClassify20 <- classify_model(container, rfModel20)

#EVALUATE THE MODELS

#NB confusion matrix
table(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)], nbClassify)
recall_accuracy(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)], nbClassify)

#MAXENT confusion matrix
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), maxentClassify[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), maxentClassify[,"MAXENTROPY_LABEL"])

#SVM confusion matrix
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), svmClassify[,"SVM_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), svmClassify[,"SVM_LABEL"])

#TREE confusion matrix
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), treeClassify[,"TREE_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), treeClassify[,"TREE_LABEL"])

#RF confusion matrix 10 random trees
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify10[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify[,"FORESTS_LABEL"])

#RF confusion matrix 20 random trees
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify20[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify[,"FORESTS_LABEL"])

#RF confusion matrix 200 random trees (default parameter)
table(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(tweetAll$polarity[(trainingSize+1):nrow(tweetAll)])), rfClassify[,"FORESTS_LABEL"])

analytics <- create_analytics(container, 
                              cbind(maxentClassify, svmClassify, treeClassify, rfClassify)) # create analytics for evaluation

execTime <- rbind(NBprocTime, maxentprocTime, svmprocTime, treeprocTime, rfprocTime) # training execution time

# save evaluation to csv files
write.csv(execTime, "ExecutionTime.csv")
write.csv(analytics@document_summary,"DocumentSummary.csv")
write.csv(analytics@algorithm_summary,"AlgorithmSummary.csv")
write.csv(analytics@ensemble_summary,"EnsembleSummary.csv")

# Save trained models
saveRDS(container, "container.rds")
saveRDS(nbModel, "nbModel.rds")
saveRDS(maxentModel, "maxentModel.rds")
saveRDS(svmModel, "svmModel.rds")
saveRDS(treeModel, "treeModel.rds")
saveRDS(rfModel, "rfModel.rds")
saveRDS(rfModel10, "rfModel10.rds")
saveRDS(rfModel20, "rfModel20.rds")

# Cross Validation
container <- readRDS("container.rds")

N <- 10 # number of folds

cross_MAXENT <- cross_validate(container, N, "MAXENT")
cross_SVM <- cross_validate(container, N, "SVM")
cross_RF <- cross_validate(container, N, "RF", ntree = 20)

saveRDS(cross_RF, "crossRF.rds")
saveRDS(cross_SVM, "crossSVM.rds")
saveRDS(cross_MAXENT, "crossMAXENT.rds")
