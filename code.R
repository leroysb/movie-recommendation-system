# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

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
edx <- edx %>% mutate(year=year(dates))

# Save both edx and validation dataset
write.csv(edx, file = "data/edx.csv")
write.csv(validation, file = "data/validation.csv")

# Number of movies and users in the edx dataset
n_distinct(edx$movieId)
n_distinct(edx$userId)

# Confirm if every user rated a movie
edx %>% filter(is.na(.$ratings))

# Top 20 most viewed genres
edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% summarize(count = n()) %>%
  top_n(20, count) %>%  arrange(desc(count))

# Top 20 most viewed movies
edx %>% group_by(movieId) %>% summarize(title = title[1], count = n()) %>% top_n(20, count) %>%  arrange(desc(count))

# What year has the highest median number of ratings
edx %>% group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>%
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Genre avarages

edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# 

edx <- mutate(movielens, date = as_datetime(timestamp))

edx %>% mutate(week = round_date(date, "week")) %>%
  group_by(week) %>% summarize(avg = mean(rating)) %>% 
  ggplot(aes(week, avg)) + geom_point() + geom_smooth()

#
movielens %>% 
  group_by(movieId) %>%
  summarize(n = n(), years = 2018 - first(year), title = title[1], rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  top_n(20, rate) %>%
  arrange(desc(rate))

edx %>% 
  group_by(movieId) %>%
  summarize(n = n(), years = 2018 - first(year), title = title[1], rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()