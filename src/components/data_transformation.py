# feature engineering, data cleaning
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #for creating pipeline to apply transformers on different columns of the data.
from sklearn.impute import SimpleImputer #handle missing values in the data.
from sklearn.pipeline import Pipeline #define a series of data transformation steps
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# file path for saving the preprocessor object
# dataclass for classes that store data attributes & automatically generates special methods like __init__
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
         This object is a pipeline that applies different transformations to numerical and categorical columns in the data.
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

           #imputer for handling missing values
           #The strategy used here is to fill missing values with the median of the corresponding column.
           # standardize the numerical columns, scales them to have a mean of 0 and a standard deviation of 1.
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            #replace all missing values 
            ##The strategy used here is to fill missing values with the mode of the corresponding column
            #convert categorical columns into one-hot encoded vectors
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # combines the numerical and categorical pipelines
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    #starts data transformation technique    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            #read train and test dataset
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            #column with target variable
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # the input features and target features are separated for both the training and testing datasets.
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            '''
            
            The fit_transform method of the preprocessing_obj is called on the training input features to fit and transform the data. 
            Then, the transform method is called on the testing input features to transform them using the already fitted preprocessing object.

            '''
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            '''

            the target features (target_feature_train_df and target_feature_test_df) are converted to NumPy arrays. 
            The np.c_ function is then used to concatenate the transformed input features with their respective target features for both the training and testing data, 
            creating the final transformed datasets train_arr and test_arr.

            '''
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)