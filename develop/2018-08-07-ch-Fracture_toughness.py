
# coding: utf-8

# In[1]:


import pandas as pd
import re
from matminer.utils.conversions import str_to_composition
from matminer.featurizers import composition
from os import environ
from citrination_client import CitrinationClient
from citrination_client import *
from pypif import pif


# # Fracture Toughness DataSet

# In[2]:


client = CitrinationClient(environ['CITRINATION_API_KEY'], 'https://citrination.com')
dataset_id = '151803'


# ## query DataSet contains Fracture-Toughness feature

# In[3]:


def parse_prop_and_temp(search_result, prop_name):
    rows = []
    pif_records = [x.system for x in search_result.hits]
    for system in pif_records:
        cryst_value = '0'
        for prop in system.properties:
            if prop.name == 'Crystallinity':
                if prop.scalars[0].value == 'Polycrystalline':
                    cryst_value = '-1'
                elif prop.scalars[0].value == 'Single Crystal':
                    cryst_value = '1'
        for prop in system.properties:
            if prop.name == prop_name:
                for cond in prop.conditions:
                    if cond.name == "Temperature":
                        if len(prop.scalars) == len(cond.scalars):
                            if cond.units=="$^{\\circ}$C":
                                add= 273.
                            elif cond.units=='K':
                                add= 0.
                            for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
                                if prop_sca.value and cond_sca.value:
                                    if cond_sca.value.isdigit():
                                        try:
                                            float(prop_sca.value)
                                            row = [system.chemical_formula, cryst_value, float(prop_sca.value), (float(cond_sca.value)+add)]
                                            rows.append(row)
                                        except ValueError as e:
                                            print(e)

    df = pd.DataFrame(rows)
    df.columns = ['Formula', 'Crystallinity', prop_name, 'Temperature (K)']
    return df


# In[4]:


prop_names = ["Fracture Toughness"]

for prop_name in prop_names:
    value_query = FieldQuery(extract_as=prop_name, extract_all=True)
    property_query = PropertyQuery(name=FieldQuery(filter=[Filter(equal=prop_name)]), value=value_query)
    formula_query = ChemicalFieldQuery(extract_as="formula")
    system_query = PifSystemQuery(chemical_formula=formula_query, properties=property_query)
    dataset_query = DatasetQuery(id=[Filter(equal=dataset_id)])
    data_query = DataQuery(dataset=dataset_query, system=system_query)
    pif_query = PifSystemReturningQuery(size=5000, random_results=False, query=data_query)

    search_result = client.search.pif_search(pif_query)

    print("We found {} records".format(len(search_result.hits)))
    print([x.extracted for x in search_result.hits[0:2]])
    df_ft = parse_prop_and_temp(search_result, prop_name)


# In[5]:


df_ft.loc[109, "Formula"]


# In[6]:


df_ft.loc[109, "Formula"] = 'ZrO.xY2O3'
print('df_ft shape: ' + str(df_ft.shape))


# ## create subDataSet of Fracture-Toughness

# In[7]:


df_ft = df_ft.groupby( ['Formula','Crystallinity','Temperature (K)'], as_index=False).mean()
df_ft.columns = ['Formula','Crystallinity', 'Temperature (K)', 'Fracture Toughness (Mpa m^(1/2))']
print('df_ft shape: ' + str(df_ft.shape))


# ### parse formula with floats

# In[8]:


# Parse the chemicalFormula, v2
# it cannot dealing leading parentheses well yet, such as  (3+x)Al2O3.(2-x)SiO2

def cleanUp(formula): #modified from Chris's function: make_chem_form_compatible
    
    for bad_str in [r'whisker', r'\(([a-z]{3,}\-?[a-z]{3,})\)']:    # delete bad strings, (solidsolution), (two-phase)
        formula = re.sub(bad_str, '', formula)
        
    formula = re.sub(r'\$(.*?)\$-', '', formula) # LaTeX expressions
    formula = re.sub(r'Cordierite', r'Mg2Fe2Al4Si5O18', formula) #chemical formular for Cordierite
    formula = re.sub(r'xSialon', r'Si2–xAlxO1+xN2–x', formula) #chemical formular for Sialon
    
    return formula

def formula_decompose(formula):
    '''
    decompose chemical formula 
    return
        composition: list, [(element,num),...]
            element: string
            num: string, can be math expression such as '1+0.5x'
    '''

    comp = []

    formula = cleanUp(formula)
    
    # recognize (prefactor)(elements)(numbers)
    # for example: 2Al2O3 is recognized as 4Al and 6O
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


# In[9]:


# create composition feature based on parsed formula
df_ft["flatFormula"] = df_ft["Formula"].map(formula_parser)
df_ft["composition"] =df_ft["flatFormula"].transform(str_to_composition)
df_ft.shape


# In[10]:


df_ft.to_csv('Fracture_Toughness_ch.csv', index=False)


# ### adding features based on compositions from matminer package

# In[11]:


# element property 
ep_feat = composition.ElementProperty.from_preset(preset_name="magpie")
df_ft = ep_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True)# input the "composition" column to the featurizer
# atomic orbitals
ao_feat = composition.AtomicOrbitals()
df_ft = ao_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True)  
# band center
bc_feat  = composition.BandCenter()
df_ft = bc_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# miedema
m_feat  = composition.Miedema()
df_ft = m_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# stoichiometry
s_feat  = composition.Stoichiometry()
df_ft = s_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# t metal fraction
tmf_feat  = composition.TMetalFraction()
df_ft = tmf_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# # valence orbital
# vo_feat  = composition.ValenceOrbital()
# df_ft = vo_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# # yang solid solution
# yss_feat  = composition.YangSolidSolution()
# df_ft = yss_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# # atomic packing efficiency
# ape_feat  = composition.AtomicPackingEfficiency()
# df_ft = ape_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 

df_ft.shape


# ### save data

# In[12]:


# save Fracture Toughness sub-dataset as csv
df_ft.to_csv('fracture_toughness_featured_ch.csv', index=False)


# # Machine Learning

# In[13]:


excluded = ['Formula', 'flatFormula', 'composition', 'Fracture Toughness (Mpa m^(1/2))','HOMO_character','HOMO_element','LUMO_character','LUMO_element','Miedema_deltaH_inter']
df_ft_1 = df_ft.drop(df_ft.index[255:258]) #drop last three rows

#data
X = df_ft_1.drop(excluded,axis=1)
X = X.dropna(axis='columns')
#target
y = df_ft_1['Fracture Toughness (Mpa m^(1/2))']


# ## linear regression model

# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))


# ### check cross validation of linear regression

# In[15]:


from sklearn.model_selection import KFold, cross_val_score

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=False, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[16]:


from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict

pf = PlotlyFig(x_title='Fracture Toughness (Mpa m^(1/2))',
               y_title='Predicted Fracture Toughness (Mpa m^(1/2))',
               title='Linear regression',
               mode='notebook',
               filename="lr_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(lr, X, y, cv=crossvalidation)), ([0, 12], [0, 12])], 
      labels=df_ft_1['Formula'], 
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )


# ## random forest model

# In[17]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# In[18]:


# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[19]:


from matminer.figrecipes.plot import PlotlyFig

pf_rf = PlotlyFig(x_title='Fracture Toughness (Mpa m^(1/2))',
                  y_title='Random forest Fracture Toughness (Mpa m^(1/2))',
                  title='Random forest regression',
                  mode='notebook',
                  filename="rf_regression.html")

pf_rf.xy([(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 12], [0, 12])], 
      labels=df_ft_1['Formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)


# In[20]:


from sklearn.model_selection import train_test_split
X['formula'] = df_ft_1['Formula']
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

