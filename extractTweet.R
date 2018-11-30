install.packages("twitteR") # if twitteR package is not yet installed
library("twitteR")

setwd("D:/DATA/Graduate/Tweet Analysis Demo/data")

consumerKey <- 'CONSUMER_KEY'
consumerSecret <- 'SECRET_KEY'
accessToken <- 'ACCESS_TOKEN_KEY'
accessSecret <- 'ACCESS_SECRET_KEY'

setup_twitter_oauth(consumerKey, consumerSecret, accessToken, accessSecret) #setup authentication
searchString <- "Duterte" # keyword query
numTweets <- 500 # number of tweets to extract
tweets = list()

tweets <- searchTwitter(searchString, n=numTweets, lang="en") # search Twitter
tweets <- unique(tweets) # save only unique data

# Create a placeholder for the file
tweetFile <- NULL

# Check if dataset.csv exists
if (file.exists("dataset.csv")){tweetFile <- read.csv("dataset.csv")}

# Merge the data in the file with our new tweets
df <- do.call("rbind", lapply(tweets, as.data.frame) )
df <- rbind(df, tweetFile)

# Remove duplicates
df <- df[!duplicated(df[c("id")]), ]

# Save
write.csv(df, file = "dataset.csv", row.names = FALSE)
