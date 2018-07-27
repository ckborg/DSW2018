
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty


# In[247]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# ### Drop non-density materials

# In[223]:


# New DataFrame containing only samples with a density value
df_dens = df1[df1.isnull()['Density'] == False]


# In[224]:


df_dens.shape


# In[225]:


# Plot occurrence of features of the reduced dataset
df_dens.count().sort_values()[-17:].plot.barh()
plt.show()


# In[226]:


# Drop all columns that contain less than 300 entries 
# (crystallinity is only features left with NaN)
df_dens = df_dens.dropna(axis=1, thresh=300)


# ### Clean up columns

# #### Check if all density units are the same

# In[228]:


# Check if all density units are the same, in which case we don't need the units.
df_dens['Density-units'].unique()


# #### Fill missing crystallinity values

# In[229]:


# Determine number of polycrystalline / single crystal samples
N_polyX = df_dens[df_dens['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = df_dens[df_dens['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[230]:


#Fill NaN values in crystallinity with polycrystalline:
df_dens['Crystallinity'] = df_dens['Crystallinity'].fillna('Polycrystalline')


# #### Convert density to float

# In[231]:


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


# In[232]:


#convert the troublesome density value to a good value
df_dens.set_value(1185, 'Density', '3.16');
df_dens.loc[1185,'Density']


# In[233]:


# Convert densities to float
df_dens['Density'] = df_dens['Density'].astype(float)


# #### Remove useless columns

# In[236]:


#Drop columns that don't contain numerical or categorical data
df_dens = df_dens.drop(['licenses','names','references','Density-units','Density-conditions','Chemical Family'], axis=1)
df_dens.head()


# ### Eliminate duplicate materials

# In[238]:


N_unique_entries = len(df_dens['chemicalFormula'].unique())
print('There are just {0} unique entries in the {1} materials.'.format(N_unique_entries,df_dens.shape[0]))


# In[249]:


df_dens.sort_values('chemicalFormula').head()


# In[251]:


df_dens = df_dens.groupby(['chemicalFormula','Crystallinity'], as_index=False).mean()


# ### Composition with Chih-Hao's method

# In[253]:


df_feat = df_dens.copy()
df_feat.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'

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

df_feat["flatFormula"] = df_feat["chemicalFormula"].map(formula_parser)
df_feat["composition"] = df_feat["flatFormula"].transform(str_to_composition)


# ### Composition with Chris' method

# In[255]:


# Add second composition column (that doesn't have float stoichiometries; Chris' version)
def make_chem_form_compatible(formula):
    
    for bad_str in ['\.','x', 'y', '\+', '\-', 'z', 'w', '\%', '\^',   # individual characters
                 'Cordierite','hisker','Sialon', # certain words that show up in some formulas
                 '\$(.*?)\$',    # LaTeX expressions
                 '\((.*?)\)',    # bracketed expressions
                 '^\d{1,2}']:    # leading numbers of 1 or 2 digits
        formula = re.sub(bad_str, '', formula)
    
    return formula

# Convert chemical formulas using above function
df_feat["comp_int"] = df_feat["chemicalFormula"].transform(make_chem_form_compatible)

# Converting chemical formula to composition object using
# matminer.utils.conversions.str_to_composition
# which in turn uses pymatgen.core.composition
df_feat["comp_int"] = df_feat["comp_int"].transform(str_to_composition)


# In[256]:


df_feat.head()


# In[261]:


sample=2
print('chemical formula: ', df_feat.loc[sample,'chemicalFormula'])
print('composition :', df_feat.loc[sample,'composition'])
print('comp_int :', df_feat.loc[sample,'comp_int'])


# ### Add Additional Features with matminer

# In[134]:


df_dens.shape


# In[135]:


df_dens_add = df_dens.copy()


# In[136]:


# Add features with matminer
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df_dens_add = ep_feat.featurize_dataframe(df_dens_add, col_id="composition", ignore_errors=True)


# In[137]:


df_dens_add.shape


# In[142]:


# List of the new columns
list(set(df_dens_add.columns) ^ set(df_dens.columns))


# In[144]:


df_dens_add['avg_dev Column'].head()


# ### Which features are numerical?

# In[47]:


df1_feat.dtypes[df1_feat.dtypes!='float64']


# In[50]:


# Exploring a non-scalar feature
df1_feat[['Thermal Expansion','Thermal Expansion-conditions','Thermal Expansion-units']].dropna()


# In[62]:


df1_feat.loc[89,'Thermal Expansion-conditions']


# In[61]:


df1_feat.loc[89,'chemicalFormula']


# In[66]:


df1_feat['Compressive Strength-minimum'].dropna()


# In[77]:


df.count().hist(bins=1000)
plt.xlim(0,100)
plt.show()


# In[71]:


plt.figure(figsize=(8,16))
df.count().sort_values()[-50:-8].plot.barh()
plt.show()

