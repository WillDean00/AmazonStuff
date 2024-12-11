library(tidymodels)
library(embed)
library(vroom)


train <- vroom("./train.csv")
test <- vroom("./test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

rf_model <- rand_forest(mtry = 1,
                        min_n = 16,
                        trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model) %>% 
  fit(data = train)

#grid_of_tuning_parameters_rf <- grid_regular(mtry(range = c(1,10)),
 #                                            min_n(),
  #                                           levels = 5)
#folds_rf <- vfold_cv(train, v = 5, repeats = 2)

#CV_results_rf <- rf_wf %>% 
 # tune_grid(resamples = folds_rf,
  #          grid = grid_of_tuning_parameters_rf,
   #         metrics = metric_set(roc_auc))


#bestTune_rf <- CV_results_rf %>% 
 # select_best(metric = "roc_auc")

#final_wf_rf <-
 # rf_wf %>%
  #finalize_workflow(bestTune_rf) %>%
  #fit(data=train)

predict_rf <- predict(rf_wf, new_data = test, type = "prob")

kaggle_submission_knn <- predict_rf %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission_knn, file = "./amazon_RF.csv" , delim = ",")
