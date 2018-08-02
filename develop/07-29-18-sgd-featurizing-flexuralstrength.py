
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


# In[3]:

#check if the entry with B-N instead of BN will be a problem for flexural strength. answer: it doesn't have a flexural strength
df.loc[3892, 'Flexural Strength']


# 
# 

# ## Looking at flexural strength

# In[36]:

# New DataFrame containing only samples with a flexural strength value
fs = df[df.isnull()['Flexural Strength'] == False]
fs.shape


# In[37]:

# Check if all units are the same, in which case we don't need the units.
fs['Flexural Strength-units'].unique()


# In[38]:

# Plot occurrence of features of the reduced dataset
fs.count().sort_values()[-17:].plot.barh()
plt.show()


# In[39]:

fsn = fs['Flexural Strength'].dropna()
fsn = pd.to_numeric(fsn, errors='coerce')
fsn.hist(bins=100)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[40]:

#we see above that one value is a huge outlier compared to all the others, which have a nice spread


# In[41]:

# Drop all columns that contain less than 400 entries 
fs = fs.dropna(axis=1, thresh=400)
fs.head()


# In[42]:

# Determine number of polycrystalline / single crystal samples
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[43]:

#Fill NaN values in crystallinity with polycrystalline:
fs['Crystallinity'] = fs['Crystallinity'].fillna('Polycrystalline')
N_polyX = fs[fs['Crystallinity']=='Polycrystalline']['Crystallinity'].shape
N_singleX = fs[fs['Crystallinity']=='Single Crystal']['Crystallinity'].shape
print('Polycrystalline: {0}, Single crystal: {1}'.format(N_polyX, N_singleX))


# In[44]:

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


# In[45]:

#examine the troublesome entry
fs.iloc[295]


# In[46]:

#fix the troublesome entry
fs.set_value(2451, 'Flexural Strength', '359');
fs.loc[2451]


# In[47]:

#convert flex strength to a float and sort by it to confirm it is behaving properly as a float
fs['Flexural Strength'] = fs['Flexural Strength'].astype(float)
fs = fs.drop(['licenses','names','references','Flexural Strength-units',], axis=1)
fs.sort_values(by='Flexural Strength')


# In[48]:

N_unique_entries = len(fs['chemicalFormula'].unique())
print('There are just {0} unique entries in the {1} materials.'.format(N_unique_entries,fs.shape[0]))


# In[49]:

#reset the index from the original data set to this reduced data set
fs = fs.reset_index()
fs.head()


# In[50]:

#we get rid of the last row because the flexural strength is a huge outlier
fs.sort_values(by='Flexural Strength')
fs = fs.drop(fs.index[[427]])
fs.sort_values(by='Flexural Strength')


# In[51]:

#look at the disribution of the data with the outlier removed
fsn1 = fs['Flexural Strength']
fsn1 = pd.to_numeric(fsn1, errors = 'coerce')
fsn1.hist(bins=50)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[52]:

#look at the duplicated entries by chemical formula. some chemicals are repeated only twice, others are repeated tens of times
fsd = fs[fs.duplicated('chemicalFormula', keep = False)]
fsd.groupby('chemicalFormula').count()


# In[53]:

#Examine the repeated values for SiC
sic = fsd.loc[fsd['chemicalFormula'] == 'SiC']
sic


# In[75]:

sic.iloc[14]['Flexural Strength-conditions']


# In[76]:

#Plot the repeated values for SiC
sic = fsd.loc[fsd['chemicalFormula'] == 'SiC']
sic1 = sic['Flexural Strength']
sic1.hist(bins=20)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[77]:

#Plot the repeated values for Si3N4
sin = fsd.loc[fsd['chemicalFormula'] == 'Si3N4']
sin1 = sin['Flexural Strength']
sin1.hist(bins=20)
plt.xlabel('Flexural Strength')
plt.ylabel('# of samples')
plt.show()


# In[78]:

#initialize temp column
fs['Temperature'] = np.nan
fs.head()


# In[80]:

#add values temperature column taken from flexural strength conditions
for mat in fs.index:
    for cond in fs.loc[mat]['Flexural Strength-conditions']:
        if cond['name']=='Temperature' and cond['units']=='$^{\\circ}$C' :
            fs.at[mat, 'Temperature']= cond['scalars'][0]['value']
fs.head()


# In[81]:

fs.isnull().sum()
#181 out of the 483 entries do not have a temperature in celcius. 167 do not have a temperature at all


# In[82]:

fs = fs.drop(['Flexural Strength-conditions',], axis=1)
fs.head()


# In[83]:

fs['Temperature'] = fs['Temperature'].fillna(23)
fs


# In[84]:

fsg = fs.groupby(['chemicalFormula','Crystallinity', 'Temperature'], as_index=False).mean()
fsg


# In[95]:

fsg.at[8,'chemicalFormula' ] = 'Al2O3'
fsg.at[8,'chemicalFormula' ]


# In[96]:

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


# In[97]:

fsg1["flatFormula"] = fsg1["chemicalFormula"].map(formula_parser)
fsg1.dropna(axis=1).head()


# In[98]:

fsg1["composition"] =fsg1["flatFormula"].transform(str_to_composition)
fsg1.dropna(axis=1).head()


# In[99]:

print(len(fsg1['composition'].unique()))
print(len(fsg1['chemicalFormula'].unique()))


# In[114]:

#we see some issues with chemical formula and composition since there isn't the same number of unique values? maybe it's ok
pd.options.display.max_rows = 1000
fsg1
fsg1.sort_values(by = 'composition')


# In[115]:

fsg2 = fsg1[fsg1.duplicated('chemicalFormula', keep = False)]
fsg2.shape


# In[116]:

fsg2 = fsg1[fsg1.duplicated('composition', keep = False)]
fsg2.shape


# ## featurizing

# In[117]:

fs_feat = fsg1.copy() # Create new DataFrame for featurization


# In[118]:

# Add features with matminer (using the floating point composition by Chih-Hao)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
fs_feat = ep_feat.featurize_dataframe(fs_feat, col_id="composition", ignore_errors=True)


# In[119]:

fs_feat.shape


# In[120]:

# List of the new columns
list(set(fs_feat.columns) ^ set(fs.columns))[:10]


# In[121]:

fs_feat.head()


# In[122]:

# Adds a column for every element. Values are fraction of atoms that belong to that element.
ep_frac = ElementFraction()
fs_feat = ep_frac.featurize_dataframe(fs_feat, col_id = "composition", ignore_errors = True)
fs_feat.head()


# In[ ]:




# In[ ]:




# In[ ]:



