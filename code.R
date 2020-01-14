# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

# set.seed(1,sample.kind="Rounding")
# if using R 3.5 or earlier, use `
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
names(edx)
dim(edx)

names(temp)
str(temp)

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

names(validation)

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Add a year column generated from timestamp
dates <- as.Date(as.POSIXct(edx$timestamp, origin="1970-01-01"))
edx <- edx %>% mutate(year=year(dates), month=month(dates))
names(edx)

# Save both edx and validation dataset
write.csv(edx, file = "data/edx.csv")
write.csv(validation, file = "data/validation.csv")


# Number of movies and users in the edx dataset
n_distinct(edx$movieId)
n_distinct(edx$userId)

# Confirm if every user rated a movie
edx %>% filter(is.na(.$ratings))

# Top 20 most viewed genres
edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% 
  summarize(count = n()) %>% top_n(20, count) %>%  arrange(desc(count))

edx %>% group_by(genres) %>% summarize(count = n()) %>% 
  top_n(20, count) %>%  arrange(desc(count))

# Top 20 most viewed movies
edx %>% group_by(movieId) %>% summarize(title = title[1], count = n()) %>% 
  top_n(20, count) %>%  arrange(desc(count)) %>% select(-movieId) %>% as.data.frame()

# What year has the highest median number of ratings
edx %>% group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>%
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Genre avarages
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# 
edx <- mutate(edx, date = as_datetime(timestamp))

edx %>% mutate(week = round_date(date, "week")) %>%
  group_by(week) %>% summarize(avg = mean(rating)) %>% 
  ggplot(aes(week, avg)) + geom_point() + geom_smooth()

# Let's look at some of the general properties of the data to better understand the challenge.
# The first thing we notice is that some movies get rated more than others.
# Here's the distribution. A normal distribution
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# A second observation is that some users are more active than others at rating movies.
# Notice that some users have rated over 1,000 movies while others have only rated a handful.
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Create a test set to assess the accuracy of the models we implement,
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To make sure we don't include users and movies in the test set that do not appear in the training set, we removed these using the semi_join function,
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# To compare different models or to see how well we're doing compared to some baseline, we need to quantify what it means to do well.
# We need a loss function based on the residual mean squared error on a test set since we can interpret it as similar to standard deviation.
# It is the typical error we make when predicting a movie rating.

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Now we're ready to build models and compare them to each other.

# We're going to predict the same rating for all movies, regardless of the user and movie.
# So what number should we predict? We can use a model-based approach.
# We know that the estimate that minimizes the residual mean squared error is the least squares estimate of mu.
# in this case, that's just the average of all the ratings.
mu_hat <- mean(train_set$rating)
mu_hat

# Compute the residual mean squared error on the test set data.
# So we're predicting all unknown ratings with this average.
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Because as we go along we will be comparing different approaches, we're going to create a table that's going to store the results that we obtain as we go along.
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# We can see that these estimates vary substantially, not surprisingly. Some movies are good. Other movies are bad.

# Remember, the overall average is about 3.5. So a b i of 1.5 implies a perfect five-star rating.
# Now let's see how much our prediction improves once we predict using the model that we just fit.
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

# Our residual mean squared error did drop a little bit.
# We already see an improvement. Now can we make it better?

# How about users? Are different users different in terms of how they rate movies? 
# To explore the data, let's compute the average rating for user, u, for those that have rated over 100 movies.
# histogram of those values

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Note that there is substantial variability across users, as well. Some users are very cranky.
# And others love every movie they watch, while others are somewhere in the middle.
# Implement user variability to our model which is b_u
# So now if a cranky user--this is a negative b_u--rates a great movie, 
# which will have a positive b_i, the effects counter each other,
# and we may be able to correctly predict that this user gave a great movie a three rather than a five, which will happen.
# And that should improve our predictions.
# So how do we fit this model? Again, we could use lm [lm(rating ~ as.factor(movieId) + as.factor(userId))]
# But again, we won't do it, because this is a big model. It will probably crash our computer.
# Instead, we will compute our approximation by computing the overall mean, u-hat, 
# the movie effects, b-hat_i, and then estimating the user effects, b_u-hat,
# by taking the average of the residuals obtained after removing the overall mean and the movie effect from the ratings y_u,i.
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Now we can see how well we do with this new model by predicting values and computing the residual mean squared error.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()
# We see that now we obtain a further improvement.
# Our residual mean squared error dropped down to about 0.88. This is the score that Netflix awarded the competition winner.


