# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:00:42 2024

@author: vwgei
"""

# To do - Compare to GMI - PDP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

# For sanity checking
from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

from scipy.stats import norm
from scipy.stats import kurtosis

from joblib import dump, load

# import pysplit

# These packages are used when the code to plot the decision tree is uncommented
# See **TREE PLOTTING** section and un-uncomment the code there
from sklearn.tree import plot_tree
from sklearn.tree import export_text

# Define a function that calcualte error metrics from predicted and actual values
def reg_model_metrics(actual,pred):
    MSE = mean_squared_error(actual,pred)
    RMSE = np.sqrt(MSE)
    actual_mean = np.mean(actual)
    RRMSE = 100*RMSE/actual_mean
    MAE = mean_absolute_error(actual, pred)
    R2 = r2_score(actual,pred)
    D2 = d2_absolute_error_score(actual, pred)
    MAXErr = max_error(actual, pred)
    EVS = explained_variance_score(actual, pred)
    return MSE, RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS 

# Function to calculate and organize metrics
def calculate_and_organize_metrics(actual, pred, t):
     
    # Calculate standard error metrics
    mse, rmse, rrmse, mae, r2, d2, max_err, evs = reg_model_metrics(actual, pred)
    
    # Organize metrics into a dictionary
    metrics_dict = {
        "Timestep":t,
        "MSE": mse,
        "RMSE": rmse,
        "RRMSE": rrmse,
        "MAE": mae,
        "R2": r2,
        "D2": d2,
        "MaxErr": max_err,
        "EVS": evs,
    }
    
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
   
    return metrics_df


def scatter_plot(actual, pred, title, var_of_int, time, unit_label, stri, comp_flag):
    # Color Matching for clarity
    if var_of_int == 'CH4':
        color = 'blue'
        cm = 'Blues'
    elif var_of_int == 'DMS':
        color = 'purple'
        cm = 'Purples'
    elif var_of_int == 'CO':
        color = 'cyan'
        cm = 'YlGnBu'
    elif var_of_int == 'O3':
        color = 'gold'
        cm = 'YlOrBr'
    elif var_of_int == 'Benzene':
        color = 'red'
        cm = 'Reds'
    elif var_of_int == 'CH3Br':
        color = 'mediumseagreen'
        cm = 'Greens'
    elif var_of_int == 'Ethane':
        color = 'orange'
        cm = 'Oranges'
    else:
        color = 'gray'
        cm = 'Greys'
        
    if comp_flag:
        cm = "Greys"
    
    MSE, RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS = reg_model_metrics(actual, pred)
        
    fig,ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, pred, edgecolors=(0,0,0), c=color, cmap=cm)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    text = r"R2 = %.2f" % (R2); text += "\n";
    text += r"D2 = %.2f" % (D2); text += "\n";
    text += r"MAE = %.2f" % (MAE); text += "\n";
    text += r"MSE = %.2f" % (MSE); text += "\n";
    text += r"RMSE = %.2f" % (RMSE);     
    plt.annotate(text, xy=(0.01, 0.9), xycoords='axes fraction',color='black', fontsize=10,bbox=dict(facecolor='none', edgecolor='none'))
    ax.set_xlabel('Measured ' + var_of_int + " " + time + " " + unit_label)
    ax.set_ylabel('Predicted ' + var_of_int + " " + time + " " + unit_label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/Scatter_" +  var_of_int + "_" + time + stri + ".png")
    plt.show()

    
    # fig,ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(actual, pred, edgecolors=(0,0,0), c=color, cmap=cm)
    # ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    # text = r"W_R2 = %.2f" % (WR2); text += "\n"; text += r"W_MAE = %.2f" % (WMAE); text += "\n"; text += r"W_MSE = %.2f" % (WMSE); text += "\n"; text += r"W_RMSE = %.2f" % (WRMSE);      
    # plt.annotate(text, xy=(0.05, 0.85), xycoords='axes fraction',color='black', fontsize=10,
    #              bbox=dict(facecolor='none', edgecolor='none'))
    # ax.set_xlabel('Measured ' + str.split(var_of_int)[0] + time + " " + label)
    # ax.set_ylabel('Predicted ' + str.split(var_of_int)[0] + time + " " + label)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title(title)
    # plt.savefig("C:/Users/vwgei/Documents/PASTEL/plots/" + m + "/Scatter_" + m + sl + str.split(var_of_int)[0] + time + stri + ".png")
    # plt.show()
    
# Permutation importance calclation and plot
def plot_permutation_importance_scores(input_model, input_x, input_y, title):
    xl = input_x.columns # input_x labels
    yl = str.split(input_y.columns[0])[0] # #this line for not smogn
    # yl = input_y.name # This line for smogn
    
    # Calculate the Variable Importance
    perm_imp = permutation_importance(input_model, input_x, input_y, n_repeats=10, random_state=rs, n_jobs=-1)
    # Sort inices
    sorted_idx = perm_imp.importances_mean.argsort()

    # Plot a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(perm_imp.importances[sorted_idx].T,
               vert=False,
               labels=xl[sorted_idx])
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/Importances_" + yl + ".png")
    plt.show()
    plt.clf()
    return perm_imp, sorted_idx

#------------------------------------------------------------------------------
# Specify Random State
rs = 76543 #187 #76543 #9876 #76543#

GMI_flag = False

tree_plotting = False

times = [str('0'),str('-1'),str('-2'),str('-3'),str('-4'),str('-5'),str('-6'),str('-7'),str('-8'),str('-9'),str('-10'),str('-11'),str('-12'),str('-13'),str('-14'),str('-15'),str('-16'),str('-17'),str('-18'),str('-19'),str('20'),str('-21'),str('-22'),str('-23'),str('-24')]

# Read in CSV file with all of the data we need. Meteorology variables + Pathdata + VOC data
file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\A_NEW_ERA5_PASTEL.csv"

# Load data from CSV into a pandas DataFrame.
data = pd.read_csv(file_path)

# Replace WAS nodata value with np.nan for consistency
data.replace(-8888, np.nan, inplace=True)

# Individual VOCs
dms = data['WAS_DMS_WAS'] 
ethane = data['WAS_Ethane_WAS'] # GMI 92
# isoprene = data['WAS_Isoprene_WAS']
benzene = data['WAS_Benzene_WAS']
# toluene = data['WAS_Toluene_WAS']
ozone = data['UCATS-O3_O3_UCATS'] #GMI 74
methane = data['NOAA-Picarro_CH4_NOAA'] # GMI 59
ch3br = data['WAS_CH3Br_WAS'] # GMI 76
h1211 = data['WAS_H1211_WAS'] # GMI 87
h1301 = data['WAS_H1301_WAS'] # GMI 88
CO = data['NOAA-Picarro_CO_NOAA']

traj_data = data.iloc[:,201:]

# WAS_CH3Br_WAS
# vocs = [ 'NOAA-Picarro_CH4_NOAA','WAS_CH3Br_WAS', 'UCATS-O3_O3_UCATS', 'NOAA-Picarro_CO_NOAA','WAS_Ethane_WAS', 'WAS_DMS_WAS']

vocs = ['WAS_Ethane_WAS']

# base_prediction_vars = data.iloc[:, np.r_[201:211,213:220]]

#------------------------------------------------------------------------------
error_metric_columns = ['RMSE', 'R2']

# Initialize a dictionary to store distributions for each error metric
error_distributions = {col: [] for col in error_metric_columns}

#------------------------------------------------------------------------------

Metrics_table_test = pd.DataFrame()
Metrics_table_train = pd.DataFrame()

# Empty list to store extracted label substrings
# substrings = []

num_runs = 1
for ii in range(num_runs):
    print("Run " + str(ii+1) + " of " + str(num_runs))
    for voc in vocs:
        trajdata_column_startpoint1 = 585 #585 #569  #201 #211
        trajdata_column_startpoint2 = 597 #597 #581  #213 #217
        
        labelparts = voc.split('_')
        # Extract the desired substring (assuming it's the second part after splitting)
        if len(labelparts) >= 2:
            label = labelparts[1]
            
        for i in range(1):
            var_of_int = voc
            
            print("In Loop")
            
            # Match unit labels to VOC
            if var_of_int == 'NOAA-Picarro_CH4_NOAA':
                unit_label = "ppbv"
                GMI_flag = True
            elif var_of_int == 'NOAA-Picarro_CO_NOAA':
                unit_label = "ppbv"
                GMI_flag = True
            elif var_of_int == 'UCATS-O3_O3_UCATS':
                unit_label = "ppbv"
                GMI_flag = True
            else:
                unit_label = "pptv"
                if var_of_int == 'WAS_Ethane_WAS':
                    GMI_flag = True
                if var_of_int == 'WAS_CH3Br_WAS':
                    GMI_flag = True
                    
    
            
            # Gets rid of bad values in the predicted variable
            data_subset = data.dropna(subset = var_of_int) 
            inverse_subset = data[~data.index.isin(data_subset.index)]
            
            data_subset = data_subset.dropna(subset = "Pressure_0")
            inverse_subset = inverse_subset.dropna(subset = "Pressure_0")
            
            columns = data_subset.columns
            
            predictionBaseVars = data_subset.iloc[:, np.r_[trajdata_column_startpoint1:trajdata_column_startpoint1+10,trajdata_column_startpoint2]]#:trajdata_column_startpoint2+4]]
            
            inversePredictionBaseVars = inverse_subset.iloc[:, np.r_[trajdata_column_startpoint1:trajdata_column_startpoint1+10,trajdata_column_startpoint2]]
            
            #----------------------------------------------------------------------
            
            prediction_var = data_subset.loc[:, [var_of_int]]
            
            PASTEL_EST = RandomForestRegressor(n_estimators=2000,criterion='absolute_error',max_depth=None,min_samples_split=2,min_samples_leaf=1, max_features=.8,bootstrap=True, oob_score=True, min_weight_fraction_leaf=0.02,verbose=0,n_jobs=10)
            
            # PASTEL_EST = RandomForestRegressor(n_estimators=2000,criterion='absolute_error',max_depth=None,min_samples_split=2,min_samples_leaf=1, max_features=.8,bootstrap=True, oob_score=True, min_weight_fraction_leaf=0.006,verbose=0,n_jobs=10)
    
                            
            # param_grid = {'n_estimators': [200],
            #                 'max_features': [.3],
            #                 'max_depth': [None], #7
            #                 'min_samples_split': [2], #3
            #                 'min_samples_leaf': [1], #2
            #                 'criterion': ['absolute_error'],
            #                 'bootstrap': [True],
            #                 'oob_score': [True],
            #                 'min_weight_fraction_leaf': [0.02],
            #                 'ccp_alpha': [0.00]
            #                 }
            
            
            # # Create the GridSearchCV object
            # grid_search = GridSearchCV(estimator=PASTEL_est, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=15)
            
            # -----------------------------------------------------------------
            x_train, x_test, y_train, y_test = train_test_split(predictionBaseVars, prediction_var, test_size=0.3, random_state=None)
            
            # DEBUG
            # print(x_train)
            
            
            # ----------------------- Calculate Sample Weights ---------------- 
            # # Dataset statistics
            mean = y_train.mean()
            std = y_train.std()
            num_samples = y_train.count()
            y_train_max = y_train.max()
            y_train_min = y_train.min()
            
            y_train_range = y_train_max - y_train_min
            
            samp_weights_train = 1 / (0.1 + (abs((y_train_max - y_train) / y_train_range)))
            
            # Define the desired range
            new_min = 1
            new_max = 10
    
            # Initialize MinMaxScaler with the desired range
            scaler = MinMaxScaler(feature_range=(new_min, new_max))
            
            # Fit and transform the DataFrame
            samp_weights_normalized = pd.DataFrame(scaler.fit_transform(samp_weights_train), columns=samp_weights_train.columns)
            
            # Plot the sample weights
            plt.figure(figsize=(10, 6))
            plt.bar(y_train.values.flatten(), samp_weights_normalized.values.flatten(), width=4, color='blue')
            plt.xlabel('Sample Value')
            plt.ylabel('Sample Weight')
            plt.title('Sample Weights for ' + label)
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            samp_weights_normalized = abs(samp_weights_train.values.flatten())
                   
            # Dataset statistics
            # Dataset statistics
            
            # x = np.linspace(y_train_min, y_train_max, len(y_train))
            # samp_weights_train = 1 / norm.pdf(x, loc=np.mean(y_train), scale=np.std(y_train))
            
            # # Normalize weights to the desired range
            # new_min = 1
            # new_max = 1000
            # scaler = MinMaxScaler(feature_range=(new_min, new_max))
            # samp_weights_normalized = pd.DataFrame(scaler.fit_transform(samp_weights_train.reshape(-1, 1)), columns=['weight'])
            
            # # Plot the sample weights
            # plt.figure(figsize=(10, 6))
            # plt.bar(y_train.values.flatten(), samp_weights_normalized.values.flatten(), color='blue')
            # plt.xlabel('Sample Value')
            # plt.ylabel('Sample Weight')
            # plt.title('Sample Weights for Each Sample')
            # plt.xticks(rotation=45)
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            # plt.tight_layout()
            # plt.show()
            
            # # Extract absolute values of normalized weights
            # samp_weights_normalized = abs(samp_weights_normalized.values.flatten())
            # --------------------------------------------------------------------
                    
            # # Initialize sample weights
            # fit_params = {'sample_weight': samp_weights_normalized}
            
            # # Fit the model to the training data
            # print("Start grid fit")
            # grid_search.fit(x_train, y_train.values.ravel(),**fit_params)
            # print("Finish grid fit")
            
            # # Print out 
            # print(var_of_int)
            # print(i+-24)
            # bestParam = grid_search.best_params_
            # # print(bestParam)
            
            # PASTEL_EST = grid_search.best_estimator_
            
            # tempcsv = pd.DataFrame(grid_search.cv_results_)
            # tempcsv.to_csv("C:/Users/vwgei/Documents/PASTEL/data/gridSearchCV_Results/GridSearchResults" + var_of_int + ".csv", index=False)
            
            # Define a custom scoring function or use built-in scoring metrics
            scoring = {'neg_mean_absolute_error': 'neg_mean_absolute_error'}  # Using negative mean squared error as scoring
            
            # # Perform cross-validation with the defined scoring metric
            # cv_results = cross_validate(PASTEL_EST, x_train, y_train.values.ravel(), cv=5, scoring='r2')
            
            # # Print the cross-validated R^2 scores
            # print("Cross-validated R^2 scores:", cv_results)
            # print("Mean R^2 score:", np.mean(cv_results['test_score']))
            
            # print("Best model fit")
            PASTEL_EST.fit(x_train, y_train.values.ravel(), samp_weights_normalized)
            
            dump(PASTEL_EST, "C:/Users/vwgei/Documents/PVOCAL/bestmodels/" + label + ".joblib")
            
            print("Start best model predictions")
            # Make predictions on the training data
            y_pred_train = PASTEL_EST.predict(x_train)
            # Make predictions on the testing data.
            y_pred_test = PASTEL_EST.predict(x_test)
            
            y_inverse_free_predict = PASTEL_EST.predict(inversePredictionBaseVars)
            y_free_predict = PASTEL_EST.predict(predictionBaseVars)
            
            #--------------------------- Entire Atom Prediction -------------------
            # Combine predictions with respective base variables into DataFrames
            FullAtomLabel = 'Modeled_' + label + '_Concentration'
            
            inverse_predicted_df = pd.DataFrame(data=np.column_stack([inversePredictionBaseVars.values, y_inverse_free_predict]),columns=inversePredictionBaseVars.columns.tolist() + [FullAtomLabel])
            
            predicted_df = pd.DataFrame(data=np.column_stack([predictionBaseVars.values, y_free_predict]),columns=predictionBaseVars.columns.tolist() + [FullAtomLabel])
            
            # Combine these DataFrames to have a modeled predicted value with every row
            combined_df = pd.concat([inverse_predicted_df, predicted_df], axis=0)
            
            # Reset index to ensure the index is sequential
            combined_df.reset_index(drop=True, inplace=True)
            combined_df.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data/" + label + "Atom.csv")
            #----------------------------------------------------------------------
            
            # Calculate and organize metrics for training set
            train_metrics = calculate_and_organize_metrics(y_train, y_pred_train, i)
            
            # Calculate and organize metrics for testing set
            test_metrics = calculate_and_organize_metrics(y_test, y_pred_test, i)
            
            # # Update distributions for each error metric
            # for col in test_metrics.columns:
            #     if (col == 'RMSE') or (col == 'R2'):
            #         error_distributions[col].append(test_metrics[col].values)
            
            # Concatenate the result to Metrics_Table
            Metrics_table_train = pd.concat([Metrics_table_train, train_metrics], ignore_index=True)
            Metrics_table_test = pd.concat([Metrics_table_test, test_metrics], ignore_index=True)
            
            # Visulize predictions on training set 
            title = 'Training Results: ' + label + " " + str(i) #str(i+-24)
            scatter_plot(y_train, y_pred_train, title, label, str(i), unit_label, "_Train",False)#, samp_weights_normalized) #str(i+-24)
    
            # Visulize predictions on testing set
            title = 'Testing Results: ' + label + " " + str(i) #str(i+-24)
            scatter_plot(y_test, y_pred_test, title, label, str(i), unit_label, "_Test", False)#, np.zeros_like(y_test.values))
            
            print('Random Forest Out of Bag Score: ')
            print(round(PASTEL_EST.oob_score_,3))
            
            #----------------------- Compare to GMI -------------------------------
            if GMI_flag:
                # Match unit labels to VOC
                if var_of_int == 'NOAA-Picarro_CH4_NOAA':
                    gmi_var = data_subset['GMI_CH4_GMI']
                elif var_of_int == 'NOAA-Picarro_CO_NOAA':
                    gmi_var = data_subset['GMI_CO_GMI']
                elif var_of_int == 'UCATS-O3_O3_UCATS':
                    gmi_var = data_subset['GMI_O3_GMI']
                elif var_of_int == 'WAS_Ethane_WAS':
                    gmi_var = data_subset['GMI_Ethane_GMI']
                else:
                    gmi_var = data_subset['GMI_CH3Br_GMI']
                    
                
                if (var_of_int == 'WAS_Ethane_WAS') | (var_of_int == 'WAS_CH3Br_WAS'):
                    gmi_var = gmi_var * 1000
                
                if (var_of_int == 'WAS_CH3Br_WAS'):
                    gmi_var = gmi_var - 4
                
                gmix_train, gmix_test, gmiy_train, gmiy_test = train_test_split(predictionBaseVars, gmi_var, test_size=0.3, random_state=rs)
                
                y_pred_concat = np.concatenate((y_pred_train, y_pred_test))
                gmi_pred_concat = np.concatenate((gmiy_train, gmiy_test))
                
                print(type(y_train))
                
                # Combine predictions with ground truth values
                ground_truth_train = y_train
                ground_truth_test = y_test
                
                # Concatenate ground truth values for train and test sets
                ground_truth_concat = pd.concat([ground_truth_train, ground_truth_test])
                
                ground_truth_concat = ground_truth_concat.squeeze(axis=1)
                
                # Create DataFrame with predictions and ground truth values
                err_df = pd.DataFrame({'Ground Truth': ground_truth_concat,
                                  'PASTEL Predictions': y_pred_concat,
                                  'GMI Predictions': gmi_pred_concat})
                
                # A) Compare errors of both models against ground truth
                # Calculate errors
                error_model1 = err_df['PASTEL Predictions'] - err_df['Ground Truth']
                error_model2 = err_df['GMI Predictions'] - err_df['Ground Truth']
                
                y_train_max = y_train.max()
                y_train_min = y_train.min()
                
                y_train_range = y_train_max - y_train_min
                
                samp_weights_train = (0.1 + abs(((y_train_max - y_train) / y_train_range)))
                
                # Define the desired range
                new_min = 1
                new_max = 1.1
    
                # Initialize MinMaxScaler with the desired range
                scaler = MinMaxScaler(feature_range=(new_min, new_max))
                
                # Fit and transform the DataFrame
                samp_weights_normalized = pd.DataFrame(scaler.fit_transform(samp_weights_train), columns=samp_weights_train.columns)
                
                # Visulize predictions on entire data set
                title = 'PASTEL vs Ground Truth ' + label + " " + str(i) #str(i+-24)
                scatter_plot(ground_truth_concat, err_df['PASTEL Predictions'], title, label, str(i), unit_label,"", True)#, np.zeros_like(ground_truth_concat))
                
                # Visulize predictions on entire data set
                title = 'GMI vs Ground Truth ' + label + " " + str(i) #str(i+-24)
                scatter_plot(ground_truth_concat, err_df['GMI Predictions'], title, label, str(i), unit_label,"", True)#, np.zeros_like(ground_truth_concat))
                
                # Create scatter plot
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                sns.scatterplot(x=err_df['Ground Truth'], y=err_df['GMI Predictions'], label='GMI Predictions')
                sns.scatterplot(x=err_df['Ground Truth'], y=err_df['PASTEL Predictions'], label='PASTEL Predictions')
                plt.xlabel('Ground Truth')
                plt.ylabel('Model Predictions')
                plt.title('Models vs Ground Truth')
                plt.legend()
                
                # B) Compare models to each other
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=err_df['PASTEL Predictions'], y=err_df['GMI Predictions'])
                plt.xlabel('PASTEL Predictions')
                plt.ylabel('GMI Predictions')
                plt.title('Comparison of PASTEL and GMI')
                
                # Show plot
                plt.tight_layout()
                plt.show()
                
                       
            # Generate permutation importances using function
            perm_importances, perm_importances_index = plot_permutation_importance_scores(PASTEL_EST, x_train, y_train, title="Permutation Importances using Train: " + label)
                
            print("perm impt indx = ")
            print(perm_importances_index)
            # Get feature importances
            feature_importances = PASTEL_EST.feature_importances_
            
            # Get indices of the two most important features
            top_two_indices = np.argsort(feature_importances)[-2:]
            top_two_permutation_indeces = perm_importances_index[-2:]
            # print()
            
            # Print the indices and names of the two most important features
            print("Feature names:", x_train.columns[top_two_indices])
            
            
            
            # Checks for partial dependence flag
            plt.clf()    
            #From SKlearn PDP tutorial!   
            top_two_feat = x_train.columns[top_two_permutation_indeces] 
            
            
            features_info_top_two = {
                "features": [top_two_feat[0], top_two_feat[1],(top_two_feat[0], top_two_feat[1])],
                "kind" : "average",
            }
            _, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
                        
            common_params = {
                "sample_weight":samp_weights_train
                }
            
            display = PartialDependenceDisplay.from_estimator(
            PASTEL_EST,
            x_train,
            **features_info_top_two,
            ax=ax,
            # **common_params,
            )
                    
            _ = display.figure_.suptitle(("1-way vs 2-way Partial Dependance for \n" + label + " " + str(i) + " of " + str(top_two_feat[0]) + " and " + str(top_two_feat[1])),fontsize=16) #str(i+-24)
            
            plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\SSE_showcase\pdp" +  label + "_" + str(i + 1) + ".png") #str(i+-24)
            plt.show()
            
            # If tree plotting flag
            if tree_plotting:
                # Visualize decision tree - Increases run time significantly
                fig = plt.figure(figsize=(30, 20))
                plot_tree(PASTEL_EST.estimators_[0], 
                          filled=True, impurity=True, 
                          rounded=True)
                fig.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\rfrtree.png")
                 
            trajdata_column_startpoint1 += 16
            trajdata_column_startpoint2 += 16
        
        # -------------------------- PLOT MISSING ATOM DATA -------------------
        # Map corners
        mapcorners = [-260, -90, 20, 90]  # Full globe
    
        ilats = inversePredictionBaseVars["Latitude_0"].values
        ilongs = inversePredictionBaseVars["Longitude_0"].values
        
        plats = predictionBaseVars["Latitude_0"].values
        plongs = predictionBaseVars["Longitude_0"].values
        
        #         
        ilongs[ilongs > 100] -= 360
        plongs[plongs > 100] -= 360
        
        # lats = inversePredictionBaseVars["Latitude_0"].values
        # longs = inversePredictionBaseVars["Longitude_0"].values
        
        # Create a map with custom boundaries
        plt.figure(figsize=(10, 8))
        map = Basemap(projection='cyl', llcrnrlon=mapcorners[0], llcrnrlat=mapcorners[1], urcrnrlon=mapcorners[2], urcrnrlat=mapcorners[3], lat_ts=20, resolution='c',lon_0=135)
        
        # Set the map boundaries
        map.drawcoastlines()
        map.drawcountries()
        
        # Draw latitude and longitude grid lines
        map.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], fontsize=10)  # Latitude lines
        map.drawmeridians(range(-180, 181, 45), labels=[0,0,0,1], fontsize=10)  # Longitude lines
        
      
        # Convert latitude and longitude to map coordinates
        ix, iy = map(ilongs, ilats)
        # Convert latitude and longitude to map coordinates
        px, py = map(plongs, plats)
        
        map.scatter(ix, iy, s=20, c=y_inverse_free_predict, cmap='cool', label='PASTEL Modeled Missing Data',edgecolor='none')
        
        plt.colorbar(label='Concentration of PASTEL Data ' + unit_label, shrink=0.6)
        
        # Plot latitude and longitude from predictionBaseVars dataframe in red
        map.scatter(px, py, s=20, c=y_free_predict, cmap='Wistia_r', label='True ATom Data',edgecolor='none')
        
        plt.colorbar(label='Concentration of ATom Data ' + unit_label, shrink=0.6)
        
        # Set labels and title
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        plt.title('PASTEL Modeled Missing ' + label + ' Data')
        
     
        # Show plot
        plt.show()
        
        #------------------ GLOBAL PREDICTIONS --------------------------------
        # Read in CSV file with all of the data we need. Meteorology variables + Pathdata + VOC data
        file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\All_Timesteps.csv"
    
        # Load data from CSV into a pandas DataFrame.
        all_time = pd.read_csv(file_path)
    
        all_time = all_time.dropna()
    
        Atom_predict = all_time.iloc[:, np.r_[0:10,12]]
    
        y_ATom_trajpoint = PASTEL_EST.predict(Atom_predict)
            
        ATom_traj_data = pd.DataFrame(data=np.column_stack([Atom_predict.values, y_ATom_trajpoint]),columns=Atom_predict.columns.tolist() + [FullAtomLabel])
    
    
        lats = ATom_traj_data["Latitude_0"].values
        longs = ATom_traj_data["Longitude_0"].values
        altitude = ATom_traj_data["AltP_meters_0"].values
        concentration = ATom_traj_data[FullAtomLabel].values
    
        ATom_traj_data.loc[ATom_traj_data["Longitude_0"] > 100, "Longitude_0"] -= 360
    
        # Map corners
        mapcorners = [-260, -90, 20, 90]  # Full globe
    
        # Create a map with custom boundaries
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlon=mapcorners[0], llcrnrlat=mapcorners[1], urcrnrlon=mapcorners[2], urcrnrlat=mapcorners[3], lat_ts=20, resolution='c',lon_0=135)
    
        # Draw coastlines, countries, and states
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
    
        # Convert latitude and longitude to map coordinates
        x, y = m(longs, lats)
    
        # Set the size of points based on altitude
        # sizes = [alt * .00005 for alt in altitude]  # Adjust multiplier to your preference
    
        # Plot points
        m.scatter(x, y, s=10, c=concentration, cmap='viridis', edgecolor='none') #viridis
    
        # Add colorbar
        plt.colorbar(label='Concentration ' + unit_label, shrink=0.6)
    
        # Draw latitude and longitude grid lines
        m.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], fontsize=10)  # Latitude lines
        m.drawmeridians(range(-180, 181, 45), labels=[0,0,0,1], fontsize=10)  # Longitude lines
    
        plt.title('Modeled ATom ' + label + ' Concentration')
        plt.show()
        
        # sns.kdeplot(concentration, color='skyblue', shade=True)
        # plt.xlabel('Concentration')
        # plt.ylabel('Density')
        # plt.title('Kernel Density Estimation of Modeled DMS Concentration')
        # plt.show()
        
        # Plot histogram
        plt.hist(concentration, bins=20, color='skyblue', edgecolor='black', label='Modeled PASTEL Distribution')
        plt.hist(prediction_var, bins=20, color='salmon', edgecolor='black', alpha=0.7, label='ATom Measured Distribution')
        plt.xlabel('Concentration of ' + label + " " + unit_label)
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison PASTEL vs ATom')
        plt.legend()
        plt.show()
        
            
    
    
        
        # time = [-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
        # # Plot each error metric over time
        # plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        
        # # Loop through each error metric
        # for column in Metrics_table_train.columns:
        #     plt.plot(time, Metrics_table_train[column], label=column)
        
        # # Add labels and title
        # plt.xlabel('Trajectory Timestamp')
        # plt.ylabel('Error Metric Value')
        # plt.title('Train Error Metrics Over Trajectory')
        
        # # Add legend
        # plt.legend()
        
        # # Show plot
        # plt.show()
        
        # for column in Metrics_table_test.columns:
        #     plt.plot(time, Metrics_table_test[column], label=column)
        
        # # Add labels and title
        # plt.xlabel('Trajectory Timestamp')
        # plt.ylabel('Error Metric Value')
        # plt.title('Test Error Metrics Over Trajectory')
        
        # # Add legend
        # plt.legend()
        
        # # Show plot
        # plt.show()




r2test = []
r2train = []

PASTEL_EST =  RandomForestRegressor(n_estimators=200,criterion='absolute_error',max_depth=None,min_samples_split=2,min_samples_leaf=1, max_features=.8,bootstrap=True, oob_score=True, min_weight_fraction_leaf=0.006,verbose=0,n_jobs=10)
# PASTEL_EST = RandomForestRegressor(n_estimators=200,criterion='absolute_error',max_depth=None,min_samples_split=2,min_samples_leaf=1, max_features=.8,bootstrap=True, oob_score=True, min_weight_fraction_leaf=0.02,verbose=0,n_jobs=10)
# sns.histplot(another_data, bins=30, kde=True, color='orange', alpha=0.5, label='Second Distribution')

print("Start Error BootStrapping Plot")
for k in range(1000):
    print("On Run :" + str(k + 1))
    x_train, x_test, y_train, y_test = train_test_split(predictionBaseVars, prediction_var, test_size=0.2, random_state=None)
    
    y_train_max = y_train.max()
    y_train_min = y_train.min()
    
    y_train_range = y_train_max - y_train_min
    
    samp_weights_train = (0.1 + abs(((y_train_max - y_train) / y_train_range)))
    
    # Define the desired range
    new_min = 1
    new_max = 1.1

    # Initialize MinMaxScaler with the desired range
    scaler = MinMaxScaler(feature_range=(new_min, new_max))
    
    # Fit and transform the DataFrame
    samp_weights_normalized = pd.DataFrame(scaler.fit_transform(samp_weights_train), columns=samp_weights_train.columns)
    
    # print("Best model fit")
    PASTEL_EST.fit(x_train, y_train.values.ravel(), samp_weights_normalized.values.ravel())
    
    # Make predictions on the testing data.
    y_pred_test = PASTEL_EST.predict(x_test)
    y_pred_train = PASTEL_EST.predict(x_train)
    
    r2test.append(r2_score(y_test,y_pred_test))
    r2train.append(r2_score(y_train,y_pred_train))
         
# Create subplots
fig, ax = plt.subplots(figsize=(8, 6)) 

test_kurtosis = kurtosis(r2test)
train_kurtosis = kurtosis(r2train)

sns.histplot(r2test, bins=30, kde=True, color='cyan', alpha=0.7)
sns.histplot(r2train, bins=30, kde=True, color='orange', alpha=0.5)

fig.text(0.1, 0.9, f'Test Kurtosis (cyan): {test_kurtosis:.2f}')
fig.text(0.1, 0.85, f'Train Kurtosis (gold): {train_kurtosis:.2f}')

ax.set_xlabel('R^2 Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Distribution of R^2 for {label}')

plt.tight_layout()
plt.show()    
    
