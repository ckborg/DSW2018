
# coding: utf-8

# In[1]:


import pandas as pd

from os import environ
from citrination_client import CitrinationClient
from citrination_client import *


# # Fracture Toughness DataSet

# In[2]:


client = CitrinationClient(environ['CITRINATION_API_KEY'], 'https://citrination.com')
dataset_id = '151803'


# ## query DataSet contains Fracture-Toughness feature

# In[3]:


value_query = FieldQuery(extract_as="Fracture Toughness", extract_all=True)
property_query = PropertyQuery(name=FieldQuery(filter=[Filter(equal="Fracture Toughness")]), value=value_query)
formula_query = ChemicalFieldQuery(extract_as="formula")
system_query = PifSystemQuery(chemical_formula=formula_query, properties=property_query)
dataset_query = DatasetQuery(id=[Filter(equal=dataset_id)])
data_query = DataQuery(dataset=dataset_query, system=system_query)
pif_query = PifSystemReturningQuery(size=5000, random_results=True, query=data_query)

search_result = client.search.pif_search(pif_query)

print("We found {} records".format(len(search_result.hits)))
print([x.extracted for x in search_result.hits[0:2]])


# In[4]:


from pypif import pif
import csv

rows = []
pif_records = [x.system for x in search_result.hits]
for system in pif_records:
    if "x" not in system.chemical_formula and "." not in system.chemical_formula:
        for prop in system.properties:
            if prop.name == "Fracture Toughness" and prop.units == "MPa m$^{1/2}$":
                for cond in prop.conditions:
                    if cond.name == "Temperature":
                        if len(prop.scalars) == len(cond.scalars):
                            for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
                                row = [system.chemical_formula, prop_sca.value, cond_sca.value, system.references[0].citation]
                                rows.append(row)


# ## create subDataSet of Fracture-Toughness

# In[25]:


df_ft = pd.DataFrame(rows)
df_ft.columns = ['Formula', 'Fracture Toughness (Mpa m^(1/2))', 'Temperature (degC)', 'Reference']
df_ft.loc[21, "Fracture Toughness (Mpa m^(1/2))"] = '3.1' #fix '3.1(10%)' to '3.1'
df_ft.loc[515, "Fracture Toughness (Mpa m^(1/2))"] = '3.1'
df_ft['Fracture Toughness (Mpa m^(1/2))'] = df_ft['Fracture Toughness (Mpa m^(1/2))'].astype(float)


# In[39]:


df_ft_2 = df_ft.groupby( ['Formula','Temperature (degC)'], as_index=False).mean()
df_ft_2.columns = ['Formula','Temperature (degC)', 'Fracture Toughness (Mpa m^(1/2))']
print('df_ft_2 shape: ' + str(df_ft_2.shape))


# ### create composition based on Formula

# In[40]:


from matminer.utils.conversions import str_to_composition
from matminer.featurizers import composition
# create composition
df_ft_2["composition"] =df_ft_2["Formula"].transform(str_to_composition)


# ### adding features based on compositions from matminer package

# In[41]:


# element property 
ep_feat = composition.ElementProperty.from_preset(preset_name="magpie")
df_ft_2 = ep_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True)# input the "composition" column to the featurizer
# atomic orbitals
ao_feat = composition.AtomicOrbitals()
df_ft_2 = ao_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True)  
# band center
bc_feat  = composition.BandCenter()
df_ft_2 = bc_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# miedema
m_feat  = composition.Miedema()
df_ft_2 = m_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# stoichiometry
s_feat  = composition.Stoichiometry()
df_ft_2 = s_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# t metal fraction
tmf_feat  = composition.TMetalFraction()
df_ft_2 = tmf_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# valence orbital
vo_feat  = composition.ValenceOrbital()
df_ft_2 = vo_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# # yang solid solution
# yss_feat  = composition.YangSolidSolution()
# df_ft_2 = yss_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 
# # atomic packing efficiency
# ape_feat  = composition.AtomicPackingEfficiency()
# df_ft_2 = ape_feat.featurize_dataframe(df_ft_2, col_id="composition", ignore_errors=True) 

df_ft_2.shape


# ### save data

# In[31]:


# save Fracture Toughness sub-dataset as csv
df_ft_2.to_csv('fracture_toughness_clean_ch.csv')


# # Machine Learning

# In[42]:


#data
X = df_ft_2.drop(['Formula','composition','Fracture Toughness (Mpa m^(1/2))','LUMO_character', 'LUMO_element','HOMO_character', 'HOMO_element','Miedema_deltaH_inter'],axis=1)
#target
y = df_ft_2['Fracture Toughness (Mpa m^(1/2))']


# ## linear regression model

# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))


# ### check cross validation of linear regression

# In[60]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=False, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# ## random forest model

# In[63]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# In[64]:


# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[70]:


from sklearn.model_selection import train_test_split
X['formula'] = df_ft_2['Formula']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
train_formula = X_train['formula']
X_train = X_train.drop('formula', axis=1)
test_formula = X_test['formula']
X_test = X_test.drop('formula', axis=1)

rf_reg = RandomForestRegressor(n_estimators=50, random_state=1)
rf_reg.fit(X_train, y_train)

# get fit statistics
print('training R2 = ' + str(round(rf_reg.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train))))
print('test R2 = ' + str(round(rf_reg.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test))))


# ### check what are the most important features used by random forest model

# In[67]:


importances = rf.feature_importances_
# included = np.asarray(included)
included = X.columns.values
indices = np.argsort(importances)[::-1]

pf = PlotlyFig(y_title='Importance (%)',
               title='Feature by importances',
               mode='notebook',
               fontsize=20,
               ticksize=15)

pf.bar(x=included[indices][0:10], y=importances[indices][0:10])

