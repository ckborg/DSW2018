
# coding: utf-8

# In[1]:

import numpy as np
import os
import matplotlib.pyplot as plt
from os import environ
from citrination_client import CitrinationClient
from citrination_client import *
from pypif import pif
from pypif.obj import *
import csv
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # Getting Flexural Strength CSV

# In[ ]:

client = CitrinationClient(environ['CITRINATION_API_KEY'], 'https://citrination.com')
dataset_id = '151803'
value_query = FieldQuery(extract_as="Flexural Strength", extract_all=True)
property_query = PropertyQuery(name=FieldQuery(filter=[Filter(equal="Flexural Strength")]), value=value_query)
formula_query = ChemicalFieldQuery(extract_as="formula")
system_query = PifSystemQuery(chemical_formula=formula_query, properties=property_query)
dataset_query = DatasetQuery(id=[Filter(equal=dataset_id)])
data_query = DataQuery(dataset=dataset_query, system=system_query)
pif_query = PifSystemReturningQuery(size=10000, random_results=False, query=data_query)
search_result = client.search.pif_search(pif_query)

print("We found {} records".format(len(search_result.hits)))
print([x.extracted for x in search_result.hits[0:5]])


# In[ ]:

rows = []
pif_records = [x.system for x in search_result.hits]
for system in pif_records:
    
    cryst_value= None
    for prop in system.properties:
        if prop.name == 'Crystallinity':
            cryst_value= prop.scalars[0].value
    for prop in system.properties:
        if prop.name == "Flexural Strength" and prop.units == "MPa":
            for cond in prop.conditions:
                if cond.name == "Temperature":
                    if cond.units=="$^{\\circ}$C":
                        add= 273.
                    elif cond.units=='K':
                        add= 0.
                    if len(prop.scalars) == len(cond.scalars):
                        for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
                            row = [system.chemical_formula, prop_sca.value, (float(cond_sca.value)+add), 
                                    cryst_value, system.references[0].citation]
                            rows.append(row)

with open('flexural_strength_temperature.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Formula', 'Flexural Strength (MPa)', 'Temperature (K)', 'Crystallinity','Reference'])
    writer.writerows(rows)


# In[ ]:

#this code is to only use the entries with no "x" or "." in the chemical formula


# rows = []
# pif_records = [x.system for x in search_result.hits]
# for system in pif_records:
#     if "x" not in system.chemical_formula and "." not in system.chemical_formula:
#         cryst_value= None
#         for prop in system.properties:
#             if prop.name == 'Crystallinity':
#                 cryst_value= prop.scalars[0].value
#         for prop in system.properties:
#             if prop.name == "Flexural Strength" and prop.units == "MPa":
#                 for cond in prop.conditions:
#                     if cond.name == "Temperature":
#                         if cond.units=="$^{\\circ}$C":
#                             add= 273.
#                         elif cond.units=='K':
#                             add= 0.
#                         if len(prop.scalars) == len(cond.scalars):
#                             for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
#                                 row = [system.chemical_formula, prop_sca.value, (float(cond_sca.value)+add), 
#                                        cryst_value, system.references[0].citation]
#                                 rows.append(row)

# with open('flexural_strength_temperature.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Formula', 'Flexural Strength (MPa)', 'Temperature (K)', 'Crystallinity','Reference'])
#     writer.writerows(rows)


# ## Clean up the raw CSV and generate composition

# In[28]:

import re
from matminer.utils.conversions import str_to_composition


# In[29]:

fs= pd.read_csv('flexural_strength_temperature.csv', encoding = "ISO-8859-1")
fs.head()


# In[30]:

len(fs['Reference'].unique())


# In[31]:

print(fs.shape)
# Determine number of polycrystalline / single crystal samples
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[32]:

#Fill NaN values in crystallinity with polycrystalline:
fs['Crystallinity'] = fs['Crystallinity'].fillna('Unknown')
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
N_unknownX = fs[fs['Crystallinity']=='Unknown']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}, Unknown: {2}'.format(N_polyX, N_singleX, N_unknownX))


# In[33]:

# Check how many values cannot simply be transformed from string to int
#also locate which row the errors occur in

N_errors, N_total, row = 0, 0, -1
for entry in fs['Flexural Strength (MPa)']:
    row+=1
    try:
        pd.Series([entry]).astype(float)
    except:
        N_errors +=1
        print(entry)
        print(row)
        
    finally:
        N_total +=1

print('{0} errors in {1} samples'.format(N_errors, N_total))


# In[34]:

fs.loc[1016, 'Reference']


# In[35]:

fs.loc[fs['Reference']=='Material Properties of a Sintered $\\alpha$-SiC, R.G. Munro, Journal of Physical and Chemical Reference Data, Vol. 26 5, pp. 1195-1203 (1997), published by American Chemical Society.']


# In[36]:

fs.at[1016, 'Flexural Strength (MPa)'] = 359
fs.loc[1016]


# In[37]:

fs['Flexural Strength (MPa)'] = fs['Flexural Strength (MPa)'].astype(float)
# fs.sort_values(by='Flexural Strength (MPa)')


# In[38]:

N_unique_entries = len(fs['Formula'].unique())
print('There are just {0} unique materials in the {1} entries.'.format(N_unique_entries,fs.shape[0]))


# In[39]:

#look at the disribution of the data 
fsn = fs['Flexural Strength (MPa)']
fsn = pd.to_numeric(fsn, errors = 'coerce')
fsn.hist(bins=100)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[40]:

fs.loc[1673, 'Reference']


# In[41]:

#remove the outliers from the dataset in case they give the machine learning models toruble
fs = fs[fs.Reference != 'Strength of Diamond, D.J. Weidner, Y. Wang, and M.T. Vaughan, Science, Vol. 266, pp. 419-421 (1994), published by American Association for the Advancement of Science.']


# In[42]:

fsn = fs['Flexural Strength (MPa)']
fsn = pd.to_numeric(fsn, errors = 'coerce')
fsn.hist(bins=50)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[43]:

fs.shape


# In[ ]:


# Parse the chemicalFormula
def formula_decompose(formula):
    '''
    decompose chemical formula 
    return
        composition: list, [(element,num),...]
            element: string
            num: string, can be math expression such as '1+0.5x'
    '''

    comp = []
    p = re.compile(r'(\d?[w-z]?)([A-Z][a-u]?)(\d*\+?\-?\d*\.?\d*[w-z]?)')

    #split the chemical formula if there is dots, but not for cases like Mg1.5x
    if re.search(r'\.', formula) and not re.search(r'\d+\.\d[w-z]', formula): 
        formula = formula.split('.')
        for item in formula:
            prefactor = '1'
            for i in re.findall(p, item):
                pre, elem, num = i
                if pre:
                    prefactor = pre
                if num == '':
                    num = '1'
                num = prefactor + '*({})'.format(num)
                comp.append((elem, num))
    else:
        prefactor = '1'
        for i in re.findall(p, formula):
            pre, elem, num = i
            if pre:
                prefactor = pre
            if num == '':
                num = '1'
            num = prefactor + '*({})'.format(num)
            comp.append((elem, num))
    return comp 

def formula_reconstruct(composition, x=0.1, y=0.1, z=0.1, w=0.1):
    '''
    reconstruct chemical formula from composition
    composition in form of [(element,num), (element,num),...]
        element: string
        num: string, can be math expression such as '1+0.5x'

    return 
        flat chemcial formula: string, such as 'Ti1.5Cu0.1Au1.0'
    '''
    flat_list = []
    for (elem, num) in composition:
        num = re.sub(r'(\d)([w-z])', r'\1*\2', num) #convert 5x to 5*x
        flat_list.append(elem)
        flat_list.append(format(eval(num), '.1f'))
    return ''.join(flat_list)
  
def formula_parser(formula):
    return formula_reconstruct(formula_decompose(formula))


# In[ ]:

fs1 = fs.copy()
fs1["flatFormula"] = fs1["Formula"].map(formula_parser)
fs1.head()
#fs1.dropna(axis=1).head()


# In[ ]:

fs1["composition"] =fs1["flatFormula"].transform(str_to_composition)
fs1.head()


# In[ ]:

print(len(fs1['composition'].unique()))
print(len(fs1['Formula'].unique()))


# ## Featurize the Dataframe

# In[ ]:

from matminer.featurizers import composition
from matminer.featurizers.composition import ElementFraction


# In[ ]:

# select featurizers

# elemental property
ep_feat = composition.ElementProperty.from_preset(preset_name="magpie")

# atomic packing efficiency
ape_feat= composition.AtomicPackingEfficiency()

# # atomic orbitals
# ao_feat = composition.AtomicOrbitals()

# stoichiometry
s_feat  = composition.Stoichiometry()

# # t metal fraction
# tmf_feat  = composition.TMetalFraction()

# # valence orbital
# vo_feat  = composition.ValenceOrbital()


# In[ ]:

fsg=fs1.drop_duplicates(['Formula','Temperature (K)','Crystallinity'])
# print('Shape of dataset with no duplicates: %s'%str(fsg.shape))
# print('Number of unique chemical Formulae in the dataset: %i'%(fsg['Formula'].nunique()))
fsg.head()


# In[ ]:

fsg= ep_feat.featurize_dataframe(fsg,col_id='composition',ignore_errors=True)
fsg= ape_feat.featurize_dataframe(fsg,col_id='composition',ignore_errors=True)
#fsg = ao_feat.featurize_dataframe(fsg, col_id="composition", ignore_errors=True)  
fsg = s_feat.featurize_dataframe(fsg, col_id="composition", ignore_errors=True) 
#fsg = tmf_feat.featurize_dataframe(fsg, col_id="composition", ignore_errors=True) 
#fsg = vo_feat.featurize_dataframe(fsg, col_id="composition", ignore_errors=True) 


# In[ ]:

fsg.to_pickle('flexural_strength_temperature_with_reduced_features.pkl')
print('Shape of dataset with Features: %s'%str(fsg.shape))


# ## Examine and Clean Up the Featurized Dataframe 

# In[2]:

#fsg= pd.read_pickle('flexural_strength_temperature_with_features.pkl')
fsg= pd.read_pickle('flexural_strength_temperature_with_reduced_features.pkl')


# In[3]:

#get rid of feature columns that have NaN 

print(fsg.isnull().values.any())
fsgd = fsg.dropna(axis=1)
print(fsgd.isnull().values.any())
print(fsgd.shape)


# In[4]:

#examine how the flexural strength depends on temp in the dataset for each material

formulae= fsgd.Formula.unique()
cfunc= plt.cm.get_cmap('hsv',len(formulae))
colors= [cfunc(i) for i in range(len(formulae))]
colors= dict(zip(formulae,colors))
symb= ['o','v','^','*','s']


plt.figure(figsize= (10,10))
for i,item in enumerate(formulae):
    item_df= fsgd[fsgd.Formula==item]
    plt.plot(item_df['Temperature (K)'].values,np.array(item_df['Flexural Strength (MPa)'],dtype= 'float'), 
             symb[i % len(symb)],color= colors[item],label= item, markersize= 5)
plt.title('Scatter Plot of Property (Flexural Strength) and Prop Condition (Temperature)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.close()


# In[5]:

# convert crystallinity to a numerical value
dict_map= {'Single Crystal':1, 'Unknown': 0, 'Polycrystalline': -1}
fsgd['Crystallinity']= fsgd['Crystallinity'].apply(lambda x: dict_map[x]) 


# In[6]:

fsgd_des= fsgd.describe()
fsgd_des= fsgd_des.astype('float')
fsgd_des


# In[7]:

#examining the std, we see that some of the columns have 0 std which is bad for ML. we will get rid of these later
std= np.array(fsgd_des.loc['std'],dtype= 'float')
std


# In[8]:

#set X and y for ML.

y = fsgd['Flexural Strength (MPa)'].astype('float')
print(y.shape)

excluded = ['Formula', 'Crystallinity', 'Reference', 'flatFormula', 'composition', 'Flexural Strength (MPa)']
X = fsgd.drop(excluded, axis=1).astype('float')
print(X.shape)

print(y.describe())

dat_des= X.describe()
dat_des


# In[9]:

# drop cols where 75%-25% < 0.5*std
drop_cols= dat_des.columns[abs(dat_des.loc['25%']-dat_des.loc['75%'])<=(0.5*dat_des.loc['std'])]
print('Number of columns to be dropped: %i'%len(drop_cols))
X= X.drop(drop_cols,axis= 1)
dat_des= dat_des.drop(drop_cols,axis= 1)
print('Shape of the training dataset: %s'%str(X.shape))


# ## Try some ML algorithms (linear regression and random forest)

# In[10]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
import numpy as np


# In[11]:

lr = LinearRegression()
lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))


# In[12]:

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[13]:

from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict

pf = PlotlyFig(x_title='Flexural Strength (MPa)',
               y_title='Predicted Flexural Strength (MPa)',
               title='Linear regression',
               mode='notebook',
               filename="lr_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(lr, X, y, cv=crossvalidation)), ([0, 12], [0, 12])], 
      labels=fsgd['Formula'], 
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )


# In[14]:

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# In[15]:

# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[16]:

from matminer.figrecipes.plot import PlotlyFig

pf_rf = PlotlyFig(x_title='Fracture Toughness (Mpa m^(1/2))',
                  y_title='Random forest Fracture Toughness (Mpa m^(1/2))',
                  title='Random forest regression',
                  mode='notebook',
                  filename="rf_regression.html")

pf_rf.xy([(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 12], [0, 12])], 
      labels=fsgd['Formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)


# In[17]:

from sklearn.model_selection import train_test_split
X['formula'] = fsgd['Formula']
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


# In[18]:

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


# In[ ]:




# In[19]:

print(included[indices][0:10])


# In[20]:

# Xr = X[['Temperature (K)', 'avg_dev NpUnfilled', 'avg_dev GSvolume_pa',
#  'avg_dev SpaceGroupNumber', 'range MeltingT', 'avg_dev MeltingT',
#  'mean GSvolume_pa', 'mean NUnfilled', 'range GSvolume_pa', 'mean NpValence']].copy()
Xr = X[['Temperature (K)', 'avg_dev NpUnfilled', 'avg_dev GSvolume_pa',
 'avg_dev SpaceGroupNumber', 'range MeltingT', 'avg_dev MeltingT', 'mean GSvolume_pa', 'mean NUnfilled' ]].copy()
#Xr = preprocessing.scale(Xr)
Xr


# In[21]:

lr = LinearRegression()
lr.fit(Xr, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(Xr, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(Xr))))


# In[22]:

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, Xr, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, Xr, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[23]:

pf = PlotlyFig(x_title='Flexural Strength (MPa)',
               y_title='Predicted Flexural Strength (MPa)',
               title='Linear regression',
               mode='notebook',
               filename="lr_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(lr, Xr, y, cv=crossvalidation)), ([0, 16], [0, 16])], 
      labels=fsgd['Formula'], 
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )


# In[24]:

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(Xr, y)
print('training R2 = ' + str(round(rf.score(Xr, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(Xr))))


# In[25]:

# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, Xr, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, Xr, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[26]:

from matminer.figrecipes.plot import PlotlyFig

pf_rf = PlotlyFig(x_title='Fracture Toughness (Mpa m^(1/2))',
                  y_title='Random forest Fracture Toughness (Mpa m^(1/2))',
                  title='Random forest regression',
                  mode='notebook',
                  filename="rf_regression.html")

pf_rf.xy([(y, cross_val_predict(rf, Xr, y, cv=crossvalidation)), ([0, 12], [0, 12])], 
      labels=fsgd['Formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)


# In[27]:

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


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# feats = {} # a dict to hold feature_name: feature_importance
# for feature, importance in zip(X.columns, rf.feature_importances_):
#     feats[feature] = importance #add the name/value pair 

# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
# imp = importances.sort_values(by = 'Gini-importance')[-10:]
# imp.plot(kind='bar')


# In[ ]:



