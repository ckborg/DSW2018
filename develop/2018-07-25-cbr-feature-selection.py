
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction


# In[68]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# # Prepare dataset

# ### Drop non-density materials

# In[27]:


# New DataFrame containing only samples with a density value
df = df[df.isnull()['Density'] == False]


# In[28]:


df.shape


# In[29]:


# Plot occurrence of features of the reduced dataset
df.count().sort_values()[-17:].plot.barh()
plt.show()


# In[30]:


# Drop all columns that contain less than 300 entries 
# (crystallinity is only features left with NaN)
df = df.dropna(axis=1, thresh=300)


# ### Clean up columns

# #### Check if all density units are the same

# In[31]:


# Check if all density units are the same, in which case we don't need the units.
df['Density-units'].unique()


# #### Fill missing crystallinity values

# In[33]:


# Determine number of polycrystalline / single crystal samples
N_polyX = df[df['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = df[df['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[34]:


#Fill NaN values in crystallinity with polycrystalline:
df['Crystallinity'] = df['Crystallinity'].fillna('Polycrystalline')


# #### Convert density to float

# In[35]:


# Check how density values cannot simply be transformed from string to int

N_errors, N_total = 0, 0
for entry in df['Density']:
    try:
        pd.Series([entry]).astype(float)
    except:
        N_errors +=1
        print(entry)
    finally:
        N_total +=1

print('{0} errors in {1} samples'.format(N_errors, N_total))


# In[36]:


#convert the troublesome density value to a good value
df.set_value(1185, 'Density', '3.16');
df.loc[1185,'Density']


# In[37]:


# Convert densities to float
df['Density'] = df['Density'].astype(float)


# #### Remove useless columns

# In[38]:


#Drop columns that don't contain numerical or categorical data
df = df.drop(['licenses','names','references','Density-units','Density-conditions','Chemical Family'], axis=1)


# ### Eliminate duplicate materials

# In[39]:


N_unique_entries = len(df['chemicalFormula'].unique())
print('There are just {0} unique entries in the {1} materials.'.format(N_unique_entries,df_dens.shape[0]))


# In[40]:


df = df.groupby(['chemicalFormula','Crystallinity'], as_index=False).mean()


# ### Composition with Chih-Hao's method

# In[41]:


df.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'

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

df["flatFormula"] = df["chemicalFormula"].map(formula_parser)
df["composition"] = df["flatFormula"].transform(str_to_composition)


# ### Composition with Chris' method

# In[42]:


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
df["comp_int"] = df["chemicalFormula"].transform(make_chem_form_compatible)

# Converting chemical formula to composition object using
# matminer.utils.conversions.str_to_composition
# which in turn uses pymatgen.core.composition
df["comp_int"] = df["comp_int"].transform(str_to_composition)


# In[43]:


df.head()


# In[44]:


sample=5
print('chemical formula: ', df.loc[sample,'chemicalFormula'])
print('composition :', df.loc[sample,'composition'])
print('comp_int :', df.loc[sample,'comp_int'])


# # Add Additional Features with matminer

# In[45]:


df_feat = df.copy() # Create new DataFrame for featurization


# ### ElementProperty

# In[46]:


df_feat.shape


# In[47]:


# Add features with matminer (using the floating point composition by Chih-Hao)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df_feat = ep_feat.featurize_dataframe(df_feat, col_id="composition", ignore_errors=True)


# In[48]:


df_feat.shape


# In[51]:


# List of the new columns
list(set(df_feat.columns) ^ set(df.columns))[:10]


# ### ElementFraction

# In[55]:


# Adds a column for every element. Values are fraction of atoms that belong to that element.
ep_frac = ElementFraction()
df_feat = ep_frac.featurize_dataframe(df_feat, col_id = "composition", ignore_errors = True)


# In[65]:


df_feat.head()


# In[66]:


df_feat['mode Column']

