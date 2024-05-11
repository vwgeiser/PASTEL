# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:02:23 2023

@author: vwgei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\A_NEW_ERA5_PVOCAL.csv"

data = pd.read_csv(file_path)

# datat0 = data.iloc[:,2:13]

# datat24 = data.iloc[:,13:26]

# dataVOC = data.iloc[:,34:134]
# 201 #211
# 213 #217

dataF = data.iloc[:, np.r_[585:595,51,52,143,146,182]]

# Replace WAS nodata value with np.nan for consistency
dataF.replace(-8888, np.nan, inplace=True)

# Plot Correlation coefficent using a heatmap
plt.figure(figsize=(20,18))
cor=dataF.corr(method='pearson')
cor = abs(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/CorrMatrixt0_pearson.png")
plt.show()



# # Plot all features and target using a pairplot
# sns.pairplot(dataF, height=2.5)
# plt.tight_layout()
# plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\ERA5_Pairplot_SSE.png")
# plt.show()

# Plot all features and target using a pairplot
# sns.pairplot(datat24, height=2.5)
# plt.tight_layout()
# plt.show()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotT24.png")

# # Plot all features and target using a pairplot
# sns.pairplot(dataVOC)
# plt.tight_layout()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotAllVOC.png")
# plt.show()