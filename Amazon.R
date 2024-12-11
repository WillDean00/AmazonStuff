#######Setting up code######

library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(themis)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = .85) %>% 
  step_smote(all_outcomes(), neighbors = 4)


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

########Logistical Regression#########

logRegModel <- logistic_reg() %>% 
  set_engine("glm")

logreg_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logRegModel) %>% 
  fit(data = train)

amazon_predictions <- predict(logreg_wf,
                             new_data=test,
                              type = "prob") 

kaggle_submission <- amazon_predictions %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission, file = "./amazon_log.csv" , delim = ",")


##########Penalized Logistical Regression#########
my_mod_pen <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
  set_engine("glmnet")

amazon_workflow_pen <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod_pen)


tuning_grid_pen <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds_pen <- vfold_cv(train, v = 10, repeats=10)

CV_results_pen <- amazon_workflow_pen %>% 
  tune_grid(resamples = folds_pen,
            grid = tuning_grid_pen,
            metrics = metric_set(roc_auc))

bestTune <- CV_results_pen %>%
select_best()

final_wf_pen <-
amazon_workflow_pen %>%
finalize_workflow(bestTune) %>%
fit(data=train)

final_wf_pen %>%
predict(new_data = train, type="prob")

amazon_pred_pen <- predict(final_wf_pen,
                           new_data = test,
                           type = "prob")

kaggle_submission <- amazon_pred_pen %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission, file = "./amazon_penalized.csv" , delim = ",")


###############KNN####################

library(tidymodels)

## knn model3
knn_model <- nearest_neighbor(neighbors=5) %>% 
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(knn_model)

knn_fit <- fit(knn_wf, data = train)

knn_predictions <- predict(knn_fit, new_data = test, type = "prob")

kaggle_submission_knn <- knn_predictions %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission_knn, file = "./amazon_knn.csv" , delim = ",")


####################Naive Bayes#######################
library(discrim)
library(naivebayes)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") 

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

tuning_grid_bayes <- grid_regular(Laplace(),
                                smoothness(),
                                levels = 3)

folds_nb <- vfold_cv(train, v = 3, repeats=3)

CV_results_nb <- nb_wf %>% 
  tune_grid(resamples = folds_nb,
            grid = tuning_grid_bayes,
            metrics = metric_set(roc_auc))

bestTune <- CV_results_nb %>%
  select_best()

final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf_nb %>%
  predict(new_data = train, type="prob")

amazon_pred_nb <- predict(final_wf_nb,
                           new_data = test,
                           type = "prob")

kaggle_submission <- amazon_pred_nb %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission, file = "./amazon_NB.csv" , delim = ",")

###########Random Forest################

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 100) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)

grid_of_tuning_parameters_rf <- grid_regular(mtry(range = c(1,10)),
                                             min_n(),
                                             levels = 5)
folds_rf <- vfold_cv(train, v = 5, repeats = 1)

CV_results_rf <- rf_wf %>% 
  tune_grid(resamples = folds_rf,
            grid = grid_of_tuning_parameters_rf,
            metrics = metric_set(roc_auc))


bestTune_rf <- CV_results_rf %>% 
  select_best(metric = "roc_auc")

final_wf_rf <-
  rf_wf %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data=train)

predict_rf <- predict(final_wf_rf, new_data = test, type = "prob")

kaggle_submission_knn <- predict_rf %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission_knn, file = "./amazon_RF.csv" , delim = ",")





