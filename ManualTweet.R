setwd("D://DATA//Graduate//Tweet Analysis R//data") # setup working directory

cleanTweet <- read.csv("realDataSet.csv", stringsAsFactors = FALSE) # read real data set file
cleanTweet <- cleanTweet[cleanTweet$polarity != "neutral", c(1, 2)] # remove neutral tweets
colnames(cleanTweet) <- c("polarity", "tweet")

# Classify tweets
matAll <- readRDS("matAll.rds")
libs <- c("SnowballC", "RTextTools")
lapply(libs, require, character.only = TRUE)

cleanTweet$tweet <- gsub("@\\w+", "", cleanTweet$tweet) # remove username
cleanTweet$tweet <- gsub("http\\S+\\s*", "", cleanTweet$tweet) # remove URL

# create document-term matrix and pre-processed data
matClassify = create_matrix(cleanTweet$tweet, language="english", originalMatrix = matAll,
                            toLower = TRUE, removeStopwords = TRUE,
                            removeNumbers = TRUE, stripWhitespace = TRUE,
                            removePunctuation = TRUE, removeSparseTerms = 0.999,
                            stemWords = TRUE, tm::weightTfIdf)
# create container for classification
containerClassify = create_container(matClassify, as.numeric(as.factor(cleanTweet$polarity)),
                                     trainSize = NULL, testSize = 1:132, virgin=FALSE)

#classify the testing data using the trained models
maxentModel <- readRDS("maxentModel.rds")
svmModel <- readRDS("svmModel.rds")
rfModel <- readRDS("rfModel.rds")

maxentClassify <- classify_model(containerClassify, maxentModel)
svmClassify <- classify_model(containerClassify, svmModel)
rfClassify <- classify_model(containerClassify, rfModel)

#MAXENT confusion matrix
table(as.numeric(as.factor(cleanTweet$polarity)), maxentClassify[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric(as.factor(cleanTweet$polarity)), maxentClassify[,"MAXENTROPY_LABEL"])

#SVM confusion matrix
table(as.numeric(as.factor(cleanTweet$polarity)), svmClassify[,"SVM_LABEL"])
recall_accuracy(as.numeric(as.factor(cleanTweet$polarity)), svmClassify[,"SVM_LABEL"])

#RF confusion matrix
table(as.numeric(as.factor(cleanTweet$polarity)), rfClassify[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(cleanTweet$polarity)), rfClassify[,"FORESTS_LABEL"])

results <- cbind(cleanTweet, maxentClassify[,"MAXENTROPY_LABEL"], svmClassify[,"SVM_LABEL"], rfClassify[,"FORESTS_LABEL"])
# save labeled results to csv
write.csv(results, "results.csv")

# create analytics for evaluation
analytics = create_analytics(containerClassify, cbind(maxentClassify, svmClassify, rfClassify))
analytics@algorithm_summary
