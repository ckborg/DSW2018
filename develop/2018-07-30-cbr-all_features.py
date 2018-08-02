
# coding: utf-8

# In[302]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# In[303]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# In[304]:


#Remove all unit  and condition columns
list_unit_cols = []
for column in df.columns:
    if '-units' in column or '-conditions' in column or '-maximum' in column or '-minimum' in column:
        list_unit_cols.append(column)
 

df = df.drop(list_unit_cols, axis=1)
df.shape


# In[305]:


# Groups entries by chemical formula and counts the number of entries for each features
df_counts = df.groupby('chemicalFormula', as_index=False).count()
df_counts.head()


# In[306]:


df_counts.shape


# In[307]:


# Checking how often there is more than one entry for a feature for a given chemical formula
(df_counts > 1).sum().sort_values(ascending=False).head()


# In[308]:


# How many of the 158 features have more than one entry for any given chemical formula?
((df_counts > 1).sum() >0).sum()


# In[309]:


list(df.columns)


# In[337]:


# List of columns that contain numerical value and can easily be converted
numerical_cols = ['chemicalFormula','Axis Length','Bulk Modulus','Cell Angle','Compressive Strength','Corrosion Activtn Energy',                   'Corrosion Rate','Creep Activatn Energy','Creep Rate','Creep Rate Exponent',                   'Debye Temperature','Density','Diffusion Coefficient','Elastic Modulus','Electrical Resistivity Log',                  'Flexural Strength','Fracture Energy','Fracture Toughness','Friction Coefficient','Grain Size',                  'Grinding Rat','Gruneisen Parameter','Hardness','Log Wear Coefficient','Log Wear Rate',                  'Maximum-Use Temperature','Melting Point Temperature','Poissons Ratio','Porosity','Relative Atomic Coordinate',                  'Shear Modulus','Sound Velocity','Specific Heat','Tensile Strength','Thermal Conductivity',                  'Thermal Diffusivity','Thermal Expansion','Weibull Modulus','Weibull Strength']

non_numerical_cols = ['Chemical Family','Commercial Name','Corrosion Species','Crystal Cell System',                      'Dopant','Elasticity Tensor','Impurity','InChI','Manufacturer','Phase','SMILES','Sintering Aid',                      'Space Group','Thermal Shock Resistance','licenses','references']

categorical_cols = ['preparation','Crystallinity','Production Form','names']


# In[320]:


print(len(numerical_cols),'+',len(non_numerical_cols),'+',len(categorical_cols))


# In[ ]:


# Remove errors given in parentheses from numerical columns
def remove_par_errs(entry_str):
    try:
        entry_str = re.sub('- ', '-', entry_str)
        ret = re.sub( '\((.*?)\)', '', entry_str)
        if entry_str=='-':
            ret=np.nan
    except:
        ret = entry_str
    return ret

# Convert column to float (dropping values that have an error in parentheses)
def conv_to_float(df, column):
    return df[column].transform(remove_par_errs).astype(float)

#conv_to_float(df, 'Axis Length').dropna().head()


# In[389]:


df_num = df[numerical_cols]
for column in numerical_cols[1:]:
    df_num[column] = conv_to_float(df, column)


# In[390]:


df_num = df_num.groupby('chemicalFormula', as_index=False).mean()


# In[381]:


plt.figure(figsize=(8,16))
df_num.count().sort_values().plot.barh()
plt.show()


# In[392]:


df_num


# ### Make features from categorical data

# In[383]:


df[categorical_cols].head()


# In[384]:


df['Crystallinity']

