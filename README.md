# 521PredictionProject
The team members are:

- Jesus Castillo
- Juan Pablo Lopez Escamilla 
- Max Osone

This repo is organazied in the following way:

- The file EDA_nepal_data.ipynb contains an EDA about the Nepal Earthquake dataset. There we checked for NAs, analyzed the target variables and the predictors.
  
- In the file pipeline.ipynb we processed the data before the modelling. For this, we onehot encoded the categorical features (except 'geo_level_2_id', 'geo_level_3_id') from the raw data provided by the professor. The raw data is located in data/raw. In this path, nepal_dat.csv is the whole training dataset provided by the professor, train.csv and test.csv is the 90%-10% split we made to validate our models and defining the best one, and nepal_evaluation.csv is the dataset without the targets provided by the professor. The processed data ready for modelling is located in the data/cleaned folder.
  
- KNN.py contains the code we used to apply TargetEncoder on 'geo_level_2_id' and 'geo_level_3_id', to select the best hyperparameters of the KNN model and to evaluate it.
  
- LogisticRegression.py contains the code we used to apply TargetEncoder on 'geo_level_2_id' and 'geo_level_3_id', to select the best hyperparameters of the Logistic Regression model and to evaluate it.
  
- lightgmb.ipynb contains the code we used to apply TargetEncoder on 'geo_level_2_id' and 'geo_level_3_id', to select the best hyperparameters of the LightGBM model and to evaluate it.
  
- xgboost_target_encoder.ipynb contains the code we used to apply TargetEncoder on 'geo_level_2_id' and 'geo_level_3_id', to select the best hyperparameters of the XGBoost model and to evaluate it.
  
- Since the XGBoost was the best model, we fitted the model with the whole dataset using the best hyperparameters from xgboost_target_encoder.ipynb. The resulting model was saved in artifacts/final_model_xgb.pkl. This is the final XGBoost model we used for interpretation and for the predictions.
  
- In the file xgboost_shap.ipynb we created some visualizations with the shap library to interpret the final XGBoost model.
  
- In the file xgboost_preds.ipynb we applied the final XGBoost model to predict the damage grade in the dataset that does not contain the target. In this file we generated a csv file with the bulding id and the prediction, and we saved this file in data/preds/predictions.csv. This is the file we shared with the professor to evaluate our model.
