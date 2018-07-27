
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction


# In[2]:

df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# ### Composition with Chih-Hao's method

# In[3]:

df1 = df.copy()
df1.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'


# In[4]:

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


# In[5]:

df1["flatFormula"] = df1["chemicalFormula"].map(formula_parser)
df1.dropna(axis=1).head()


# In[6]:

df1["composition"] =df1["flatFormula"].transform(str_to_composition)
df1.dropna(axis=1).head()


# ### Drop samples without 'density'

# In[7]:

# New DataFrame containing only samples with a density value
df_dens = df1[df1.isnull()['Density'] == False]


# In[8]:

df_dens.shape


# In[9]:

# Plot occurrence of features of the reduced dataset
df_dens.count().sort_values()[-17:].plot.barh()
plt.show()


# In[10]:

# Drop all columns that contain less than 300 entries 
# (crystallinity is only features left with NaN)
df_dens = df_dens.dropna(axis=1, thresh=300)


# ### Clean up columns

# In[11]:

df_dens.head()


# We can drop some more columns that don't contain numerical or categorical data. Further, we 

# In[12]:

# Check if all density units are the same, in which case we don't need the units.
df_dens['Density-units'].unique()


# In[13]:

# Determine number of polycrystalline / single crystal samples
N_polyX = df_dens[df_dens['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = df_dens[df_dens['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[14]:

#Fill NaN values in crystallinity with polycrystalline:
df_dens['Crystallinity'] = df_dens['Crystallinity'].fillna('Polycrystalline')


# In[15]:

# Check how density values cannot simply be transformed from string to int

N_errors, N_total = 0, 0
for entry in df_dens['Density']:
    try:
        pd.Series([entry]).astype(float)
    except:
        N_errors +=1
        print(entry)
    finally:
        N_total +=1

print('{0} errors in {1} samples'.format(N_errors, N_total))





# ## Convert a column of dataframe (e.g. Density) to float values from strings

# In[17]:

df_dens.shape


# In[44]:

df_d = df_dens.drop(['licenses','names','references','Density-units','Density-conditions','Chemical Family'], axis=1)
df_d.head()


# In[45]:

# Check how density values cannot simply be transformed from string to int
#also locate which row the errors occur in

N_errors, N_total, row = 0, 0, -1
for entry in df_d['Density']:
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


# In[46]:

#examine the troublesome density value in the 109th row of df_d
df_d.iloc[109]


# In[47]:

#convert the troublesome density value to a good value
df_d.set_value(1185, 'Density', '3.16');
df_d.loc[1185]


# In[48]:

df_d['index1'] = df_d.index
df_d.head()


# In[49]:

#convert densities to floats
#df_d['Density'].astype(float)

N_errors, N_total, row = 0, 0, -1
for entry in df_d['index1']:
    row+=1
    try:
        dens_float = float(df_d.loc[entry]['Density'])
        df_d.set_value(entry, 'Density', dens_float)
    except:
        N_errors +=1
        print(entry)
        print(row)
        
    finally:
        N_total +=1

print('{0} errors in {1} samples'.format(N_errors, N_total))


# In[51]:

#sort by density
df_d.sort_values(by='Density')


# ## Add features with element fraction

# In[53]:

df_d_feat = df_d.copy()
ep_frac = ElementFraction()
df_d_feat = ep_frac.featurize_dataframe(df_d_feat, col_id = "composition", ignore_errors = True)


# In[54]:

df_d_feat.shape


# In[55]:

list(set(df_d_feat.columns))


# In[60]:

df_d_feat.head()


# In[58]:

#how many entires had an error? 398-395 = 3 errors
df_d_feat.dropna().shape


# In[59]:

df_d_feat.dropna().head()


# In[61]:

df_d_feat['Si']


# ### Add Additional Features with matminer

# In[ ]:

df_dens.shape


# In[ ]:

df_dens_add = df_dens.copy()


# In[ ]:

# Add features with matminer
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df_dens_add = ep_feat.featurize_dataframe(df_dens_add, col_id="composition", ignore_errors=True)


# In[ ]:

df_dens_add.shape


# In[ ]:

# List of the new columns
list(set(df_dens_add.columns) ^ set(df_dens.columns))


# In[ ]:

df_dens_add['avg_dev Column'].head()


# ### Which features are numerical?

# In[ ]:

df1_feat.dtypes[df1_feat.dtypes!='float64']


# In[ ]:

# Exploring a non-scalar feature
df1_feat[['Thermal Expansion','Thermal Expansion-conditions','Thermal Expansion-units']].dropna()


# In[ ]:

df1_feat.loc[89,'Thermal Expansion-conditions']


# In[ ]:

df1_feat.loc[89,'chemicalFormula']


# In[ ]:

df1_feat['Compressive Strength-minimum'].dropna()


# In[ ]:

df.count().hist(bins=1000)
plt.xlim(0,100)
plt.show()


# In[ ]:

plt.figure(figsize=(8,16))
df.count().sort_values()[-50:-8].plot.barh()
plt.show()

