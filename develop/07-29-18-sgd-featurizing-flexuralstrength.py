
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


# In[30]:

df.loc[3892, 'Flexural Strength']


# In[3]:

# df1 = df.copy()
# df1.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'


# In[4]:

# # Parse the chemicalFormula
# def formula_decompose(formula):
#     '''
#     decompose chemical formula 
#     return
#         composition: list, [(element,num),...]
#             element: string
#             num: string, can be math expression such as '1+0.5x'
#     '''

#     comp = []
#     p = re.compile(r'(\d?[w-z]?)([A-Z][a-u]?)(\d*\+?\-?\d*\.?\d*[w-z]?)')

#     #split the chemical formula if there is dots, but not for cases like Mg1.5x
#     if re.search(r'\.', formula) and not re.search(r'\d+\.\d[w-z]', formula): 
#         formula = formula.split('.')
#         for item in formula:
#             prefactor = '1'
#             for i in re.findall(p, item):
#                 pre, elem, num = i
#                 if pre:
#                     prefactor = pre
#                 if num == '':
#                     num = '1'
#                 num = prefactor + '*({})'.format(num)
#                 comp.append((elem, num))
#     else:
#         prefactor = '1'
#         for i in re.findall(p, formula):
#             pre, elem, num = i
#             if pre:
#                 prefactor = pre
#             if num == '':
#                 num = '1'
#             num = prefactor + '*({})'.format(num)
#             comp.append((elem, num))
#     return comp 

# def formula_reconstruct(composition, x=0.1, y=0.1, z=0.1, w=0.1):
#     '''
#     reconstruct chemical formula from composition
#     composition in form of [(element,num), (element,num),...]
#         element: string
#         num: string, can be math expression such as '1+0.5x'

#     return 
#         flat chemcial formula: string, such as 'Ti1.5Cu0.1Au1.0'
#     '''
#     flat_list = []
#     for (elem, num) in composition:
#         num = re.sub(r'(\d)([w-z])', r'\1*\2', num) #convert 5x to 5*x
#         flat_list.append(elem)
#         flat_list.append(format(eval(num), '.1f'))
#     return ''.join(flat_list)
  
# def formula_parser(formula):
#     return formula_reconstruct(formula_decompose(formula))


# In[5]:

# df1["flatFormula"] = df1["chemicalFormula"].map(formula_parser)
# df1.dropna(axis=1).head()


# In[6]:

# df1["composition"] =df1["flatFormula"].transform(str_to_composition)
# df1.dropna(axis=1).head()


# 
# 

# ## Looking at various properties in the original dataset (e.g. flexural strength)

# In[7]:

# New DataFrame containing only samples with a flexural strength value
fs = df[df.isnull()['Flexural Strength'] == False]
fs.shape


# In[8]:

# Check if all units are the same, in which case we don't need the units.
fs['Flexural Strength-units'].unique()


# In[9]:

# Plot occurrence of features of the reduced dataset
fs.count().sort_values()[-17:].plot.barh()
plt.show()


# In[10]:

fsn = fs['Flexural Strength'].dropna()
fsn = pd.to_numeric(fsn, errors='coerce')
fsn.hist(bins=100)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[11]:

# Drop all columns that contain less than 400 entries 
fs = fs.dropna(axis=1, thresh=400)
fs.head()


# In[12]:

# Determine number of polycrystalline / single crystal samples
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[13]:

#Fill NaN values in crystallinity with polycrystalline:
fs['Crystallinity'] = fs['Crystallinity'].fillna('Polycrystalline')
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[14]:

# Check how many values cannot simply be transformed from string to int
#also locate which row the errors occur in

N_errors, N_total, row = 0, 0, -1
for entry in fs['Flexural Strength']:
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


# In[15]:

fs.iloc[295]


# In[16]:

fs.set_value(2451, 'Flexural Strength', '359');
fs.loc[2451]


# In[17]:

fs['Flexural Strength'] = fs['Flexural Strength'].astype(float)
fs = fs.drop(['licenses','names','references','Flexural Strength-units','Flexural Strength-conditions'], axis=1)
fs.sort_values(by='Flexural Strength')


# In[18]:

N_unique_entries = len(fs['chemicalFormula'].unique())
print('There are just {0} unique entries in the {1} materials.'.format(N_unique_entries,fs.shape[0]))


# In[19]:

fs = fs.reset_index()
fs.head()


# In[21]:

fs.sort_values(by='Flexural Strength')


# In[25]:

#we get rid of the last row because the flexural strength is a huge outlier
fs = fs.drop(fs.index[[427]])
fs.sort_values(by='Flexural Strength')


# In[26]:

fsg = fs.groupby(['chemicalFormula','Crystallinity'], as_index=False).mean()
fsg.head()


# In[27]:

len(fs['Chemical Family'].unique())


# In[31]:

fsg1 = fsg.copy()

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


# In[33]:

fsg1["flatFormula"] = fsg1["chemicalFormula"].map(formula_parser)
fsg1.dropna(axis=1).head()


# In[34]:

fsg1["composition"] =fsg1["flatFormula"].transform(str_to_composition)
fsg1.dropna(axis=1).head()


# In[36]:

print(len(fsg1['composition'].unique()))
print(len(fsg1['chemicalFormula'].unique()))


# In[39]:

#we see some issues with chemical formula and composition since there isn't the same number of unique values
pd.options.display.max_rows = 1000
fsg1


# In[ ]:




# In[ ]:




# In[ ]:




# In[40]:

fs_feat = fsg1.copy() # Create new DataFrame for featurization


# In[41]:

# Add features with matminer (using the floating point composition by Chih-Hao)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
fs_feat = ep_feat.featurize_dataframe(fs_feat, col_id="composition", ignore_errors=True)


# In[42]:

fs_feat.shape


# In[43]:

# List of the new columns
list(set(fs_feat.columns) ^ set(fs.columns))[:10]


# In[45]:

# Adds a column for every element. Values are fraction of atoms that belong to that element.
ep_frac = ElementFraction()
fs_feat = ep_frac.featurize_dataframe(fs_feat, col_id = "composition", ignore_errors = True)
fs_feat.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# New DataFrame containing only samples with a fracture toughness value
fs = df[df.isnull()['Fracture Toughness'] == False]
fs.shape


# In[ ]:



