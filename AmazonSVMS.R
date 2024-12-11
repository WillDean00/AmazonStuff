library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(kernlab)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = .85)


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


vmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
set_engine("kernlab")

amazon_workflow_SVMS <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(vmRadial)

tuning_grid_SVMS <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 3)

folds_SVMS <- vfold_cv(train, v = 3, repeats=3)

CV_results_SVMS <- amazon_workflow_SVMS %>% 
  tune_grid(resamples = folds_SVMS,
            grid = tuning_grid_SVMS,
            metrics = metric_set(roc_auc))

bestTune <- CV_results_SVMS %>%
  select_best()

final_wf_SVMS <-
  amazon_workflow_SVMS %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf_SVMS %>%
  predict(new_data = train, type = "prob")

amazon_pred_SVMS <- predict(final_wf_SVMS,
                           new_data = test,
                           type = "prob")

kaggle_submission <- amazon_pred_SVMS %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)

vroom_write(x = kaggle_submission, file = "./amazon_SVMS.csv" , delim = ",")

