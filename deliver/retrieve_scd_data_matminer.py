
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty


# In[2]:


# Retrieve NIST SCD dataset from Citrine using matminer.
# The data will be stored in the df DataFrame.

first_retrieve = False #change it to indicate first time retrieve dataset or not

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
from os import environ

if first_retrieve:

    api_key = environ['CITRINATION_API_KEY'] # insert your api key here
    c = CitrineDataRetrieval(api_key=api_key)
    df = c.get_dataframe(criteria={'data_set_id': '151803'})
    
    # Save downloaded dataset
    df.to_csv('NIST_CeramicDataSet.csv')
    df.to_pickle('NIST_CeramicDataSet.pkl')
else:
    df  = pd.read_pickle('NIST_CeramicDataSet.pkl')


# In[3]:


# Get the number of samples and number of features of the dataset
df.shape


# In[4]:


# Looking at the first 5 entries
df.head()


# In[6]:


# Taking a look at a sample entry
df.loc[42,:].dropna()


# In[7]:


# Plot a bar chart showing the 50 most common features
plt.figure(figsize=(8,16))
df.count().sort_values()[-50:].plot.barh()
plt.show()


# In[8]:


density = df['Density'].dropna()
density = pd.to_numeric(density, errors='coerce')
density.hist(bins=100)
plt.xlabel('Density')
plt.ylabel('# of samples')
plt.show()


# # Featurization

# In[9]:


# Create copy of original data to not mess with them
feat = df.copy()


# ### Make chemical formula compatible with pymatgen.core.composition

# In[10]:


# Initialize composition column
feat['composition'] = feat['chemicalFormula']


# In[11]:


# Check how many formulas cause an error when they are fed to pymatgen.core.composition (via str_to_composition)

N_errors, N_total = 0, 0
for entry in feat['composition']:
    try:
        pd.Series([entry]).transform(str_to_composition)
    except:
        N_errors +=1
        #print(entry)
    finally:
        N_total +=1

print('{0} errors in {1} samples'.format(N_errors, N_total))


# In[12]:


# This function removes certain characters and expressions from a chemical formula
# so that it can be converted using pymatgen.core.composition

def make_chem_form_compatible(formula):
    
    for bad_str in ['\.','x', 'y', '\+', '\-', 'z', 'w', '\%', '\^',   # individual characters
                 'Cordierite','hisker','Sialon', # certain words that show up in some formulas
                 '\$(.*?)\$',    # LaTeX expressions
                 '\((.*?)\)',    # bracketed expressions
                 '^\d{1,2}']:    # leading numbers of 1 or 2 digits
        formula = re.sub(bad_str, '', formula)
    
    return formula


# In[13]:


# Convert chemical formulas using above function
feat["composition"] = feat["composition"].transform(make_chem_form_compatible)

# Converting chemical formula to composition object using
# matminer.utils.conversions.str_to_composition
# which in turn uses pymatgen.core.composition
feat["composition"] = feat["composition"].transform(str_to_composition)


# In[14]:


feat['composition'].head()


# ### Parse chemical formula
# #### x, y, z, and w are setted to be 0.1 (can be modified)

# In[15]:


df1 = df.copy()
df1.loc[3892,'chemicalFormula'] = 'BN' #fix 'B-N' to 'BN'


# In[16]:


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


# In[17]:


df1["flatFormula"] = df1["chemicalFormula"].map(formula_parser)
df1.dropna(axis=1).head()


# In[18]:


df1["composition"] =df1["flatFormula"].transform(str_to_composition)
df1.dropna(axis=1).head()


# In[19]:


#check the composition object
df1.loc[4,'composition']


# ### Featurize with matminer

# In[20]:


df1.shape


# In[21]:


df1_feat = df1.copy()


# In[22]:


# Add new features to df_feat
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df1_feat = ep_feat.featurize_dataframe(df1_feat, col_id="composition", ignore_errors=True)


# In[23]:


df1_feat.shape


# In[24]:


# List of the new columns
list(set(df1_feat.columns) ^ set(df1))


# In[25]:


df1_feat['avg_dev MeltingT'].head()

