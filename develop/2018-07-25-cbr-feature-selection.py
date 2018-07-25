
# coding: utf-8

# In[14]:


import pandas as pd
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty


# In[15]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# ### Composition with Chih-Hao's method

# In[16]:


df1 = df.copy()
df1.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'


# In[17]:


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


# In[18]:


df1["flatFormula"] = df1["chemicalFormula"].map(formula_parser)
df1.dropna(axis=1).head()


# In[19]:


df1["composition"] =df1["flatFormula"].transform(str_to_composition)
df1.dropna(axis=1).head()


# In[22]:


df1.shape


# In[20]:


df1_feat = df1.copy()


# In[21]:


# Add features with matminer
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df1_feat = ep_feat.featurize_dataframe(df1_feat, col_id="composition", ignore_errors=True)


# In[23]:


df1_feat.shape


# In[24]:


# List of the new columns
list(set(df1_feat.columns) ^ set(df1))


# In[25]:


df1_feat['avg_dev MeltingT'].head()

