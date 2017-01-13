df = read.csv ("ismis17_trainingData/trainingData.csv", sep = ";")
test_df = read.csv ("testData.csv", sep=";")

library(gsubfn)

# return the most frequent element from a list
find_major_vote <- function(InVec) {
  names(which.max(table(InVec)))
}

# calculate accuracy of a confusion matrix
calc_acc = function (a_table) {
  sum(diag(a_table)) / sum(a_table)
}

# wait for any key
waitKey <- function()
{
  cat ("Press [enter] to continue")
  line <- readline()
}

#first try
# major voting between experts

try_1 = function (df) {
  predicts = c()
  
  for (i in 1:nrow(df)) {
    print (paste ("Processing train data - line ", i))
    x = df[i,]$Recommendation
    x = as.character(x)
    x1 = strapplyc (x, "\\{.*?\\}")[[1]]
    expert_ideas = c()
    for (j in 1:length(x1)) {
      x2 = substring(x1[[j]],2)
      x3 = substr(x2, 1, nchar(x2) - 1)
      recommendation = strsplit(x3, split = ",")
      expert_idea = recommendation[[1]][2]
      expert_ideas = c(expert_ideas, expert_idea)
    }
    expert_ideas = as.factor(expert_ideas)
    # major_vote = sort(table(expert_ideas),decreasing=TRUE)[1:3]
    pred = find_major_vote(expert_ideas)
    # print (pred)
    predicts = c(predicts, pred)
  }
  t = table (as.factor(predicts), df$Decision)
  # accuracy = 0.4
  predicts
}

# use random forest

try_2 = function (df, test_df) {
  predicts = c()
  
  # load the train data
  Buys = c()
  Holds = c()
  Sells = c()
  
  for (i in 1:nrow(df)) {
    print (paste ("Processing line ", i))
    x = df[i,]$Recommendation
    x = as.character(x)
    x1 = strapplyc (x, "\\{.*?\\}")[[1]]
    cur_buy = 0
    cur_hold = 0
    cur_sell = 0
    for (j in 1:length(x1)) {
      x2 = substring(x1[[j]],2)
      x3 = substr(x2, 1, nchar(x2) - 1)
      recommendation = strsplit(x3, split = ",")
      expert_idea = recommendation[[1]][2]
      if (expert_idea == "Hold") {
        cur_hold = cur_hold + 1
      } 
      else if (expert_idea == "Buy") {
        cur_buy = cur_buy + 1
      }
      else if (expert_idea == "Sell") {
        cur_sell = cur_sell + 1
      }
    }
    if (cur_sell + cur_buy + cur_hold != length(x1)) {
      print (cur_buy)
      print (cur_hold)
      print (cur_sell)
      print (length(x1))
      waitKey()
    }
    Buys = c(Buys, cur_buy)
    Sells = c(Sells, cur_sell)
    Holds = c(Holds, cur_hold)
  }
  
  Result = df$Decision
  df1 = data.frame(Buys, Holds, Sells, Result)
  
  # load the test data
  tBuys = c()
  tHolds = c()
  tSells = c()
  
  for (i in 1:nrow(test_df)) {
    print (paste ("Processing test data - line ", i))
    x = test_df[i,]$Recommendation
    x = as.character(x)
    x1 = strapplyc (x, "\\{.*?\\}")[[1]]
    cur_buy = 0
    cur_hold = 0
    cur_sell = 0
    for (j in 1:length(x1)) {
      x2 = substring(x1[[j]],2)
      x3 = substr(x2, 1, nchar(x2) - 1)
      recommendation = strsplit(x3, split = ",")
      expert_idea = recommendation[[1]][2]
      if (expert_idea == "Hold") {
        cur_hold = cur_hold + 1
      } 
      else if (expert_idea == "Buy") {
        cur_buy = cur_buy + 1
      }
      else if (expert_idea == "Sell") {
        cur_sell = cur_sell + 1
      }
    }
    if (cur_sell + cur_buy + cur_hold != length(x1)) {
      print (cur_buy)
      print (cur_hold)
      print (cur_sell)
      print (length(x1))
      waitKey()
    }
    tBuys = c(tBuys, cur_buy)
    tSells = c(tSells, cur_sell)
    tHolds = c(tHolds, cur_hold)
  }
  
  df2 = data.frame(tBuys, tHolds, tSells)
  colnames (df2) = c("Buys","Holds","Sells")
  
  library(h2o)
  
  h2o.init()
  
  # parameters for h2o
  predictors = 1:3
  response = 4
  nfolds = 5
  
  #random forest
  # rf1 = h2o.randomForest(x=1:3,y=4,training_frame =as.h2o(df1), ntrees = 2000)
  # predicts = h2o.predict(rf1, newdata = as.h2o(df2))
  # 
  # x = as.vector(predicts$predict)
  # write.table(x, file = "predic1.csv", col.names = FALSE, row.names = FALSE)
  
  #GBM
  hyper_params = list(
    ## restrict the search to the range of max_depth established above
    max_depth = seq(20,30,1),

    ## search a large space of row sampling rates per tree
    sample_rate = seq(0.2,1,0.01),

    ## search a large space of column sampling rates per split
    col_sample_rate = seq(0.2,1,0.01),

    ## search a large space of column sampling rates per tree
    col_sample_rate_per_tree = seq(0.2,1,0.01),

    ## search a large space of how column sampling per split should change as a function of the depth of the split
    col_sample_rate_change_per_level = seq(0.9,1.1,0.01),

    ## search a large space of the number of min rows in a terminal node
    min_rows = 2^seq(0,log2(nrow(df1))-1,1),

    ## search a large space of the number of bins for split-finding for continuous and integer columns
    nbins = 2^seq(4,10,1),

    ## search a large space of the number of bins for split-finding for categorical columns
    nbins_cats = 2^seq(4,12,1),

    ## search a few minimum required relative error improvement thresholds for a split to happen
    min_split_improvement = c(0,1e-8,1e-6,1e-4),

    ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
    histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")
  )

  search_criteria = list(
    ## Random grid search
    strategy = "RandomDiscrete",

    ## limit the runtime to 60 minutes
    max_runtime_secs = 3600,

    ## build no more than 100 models
    max_models = 100,

    ## random number generator seed to make sampling of parameter combinations reproducible
    seed = 1234,

    ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
    stopping_rounds = 5,
    stopping_metric = "misclassification",
    stopping_tolerance = 1e-3
  )

  grid <- h2o.grid(
    ## hyper parameters
    hyper_params = hyper_params,

    ## hyper-parameter search configuration (see above)
    search_criteria = search_criteria,

    ## which algorithm to run
    algorithm = "gbm",

    ## identifier for the grid, to later retrieve it
    grid_id = "final_grid",

    ## standard model parameters
    x = predictors,
    y = response,
    training_frame = as.h2o(df1),
    nfolds=5,

    ## more trees is better if the learning rate is small enough
    ## use "more than enough" trees - we have early stopping
    ntrees = 10000,

    ## smaller learning rate is better
    ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
    learn_rate = 0.05,

    ## learning rate annealing: learning_rate shrinks by 1% after every tree
    ## (use 1.00 to disable, but then lower the learning_rate)
    learn_rate_annealing = 0.99,

    ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
    max_runtime_secs = 3600,

    ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
    stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "misclassification",

    ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
    score_tree_interval = 10,

    ## base random number generator seed for each model (automatically gets incremented internally for each model)
    seed = 1234
  )

  ## Sort the grid models by AUC
  sortedGrid <- h2o.getGrid("final_grid", sort_by = "accuracy", decreasing = TRUE)
  sortedGrid
  
  h2o.shutdown(FALSE)
  
  predicts
}