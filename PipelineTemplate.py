# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:17:10 2025

@author: jesusCastillo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# %%
#Read dataframe and split into X and Y

data = pd.read_csv('train.csv')

# Separate features and target
X = data.drop(columns=['damage_grade'])
y = data['damage_grade']

categorical_features = X_categorical = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
numeric_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 'has_secondary_use_other']
# %%

# Preprocessing: OneHotEncode categorical columns, leave numeric columns as-is
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)
# %%

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# %%

#For steps only
#X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X)
#names
#feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
#All but one
#X_transformed = pipeline[:-1].fit_transform(X)
XTransformed = pipeline.fit_transform(X)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()



# %%
# Define a pipeline: preprocessing + classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])