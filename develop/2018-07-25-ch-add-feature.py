
# coding: utf-8

# In[1]:


import pandas as pd
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers import composition


# In[2]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')
df.head()


# ## Composition parser via regular expression

# In[3]:


# Parse the chemicalFormula, v2
# it cannot dealing leading parentheses well yet, such as  (3+x)Al2O3.(2-x)SiO2

def cleanUp(formula): #modified from Chris's function: make_chem_form_compatible
    
    for bad_str in ['whisker', r'\(([a-z]{3,}\-?[a-z]{3,})\)']:    # delete bad strings, (solidsolution), (two-phase)
        formula = re.sub(bad_str, '', formula)
        
    formula = re.sub(r'\$(.*?)\$-', '', formula) # LaTeX expressions
    formula = re.sub(r'Cordierite', 'Mg2Fe2Al4Si5O18', formula) #chemical formular for Cordierite
    formula = re.sub(r'xSialon', 'Si2–xAlxO1+xN2–x', formula) #chemical formular for Sialon
    
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


# ##  Manually Clean up Data Set

# In[4]:


# problematic entries
print('1185: ' + df.loc[1185,'Density'])
print('2696: ' + df.loc[2696, "chemicalFormula"])
print('3892: ' + df.loc[3892,"chemicalFormula"])
print('3863: ' + df.loc[3863, "chemicalFormula"])

# need to modify parser to take care of leading parentheses
print('2902: ' + df.loc[2902, "chemicalFormula"] + ' ------leading parentheses cannot be parsed correctly yet')


# In[5]:


# update chemicalFormula by manually clean-up
df.loc[1185,"Density"] = '3.16'
df.loc[2696,"chemicalFormula"] = 'Si3N4.xBaOY2O3'
df.loc[3892,"chemicalFormula"] = 'BN'
df.loc[3863,"chemicalFormula"] = 'C'
# update chemicalFormula with cleanUp function
df["chemicalFormula"] = df["chemicalFormula"].map(cleanUp)


# ## DataFrame with new features

# In[6]:


# create composition feature based on parsed formula
df["flatFormula"] = df["chemicalFormula"].map(formula_parser)
df["composition"] =df["flatFormula"].transform(str_to_composition)
df.shape


# ### add features based on composition (composition with floats)

# In[7]:


# element property 
ep_feat = composition.ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)# input the "composition" column to the featurizer
df.shape


# In[8]:


# atomic orbitals
ao_feat = composition.AtomicOrbitals()
df = ao_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)  
df.shape


# In[9]:


# band center
bc_feat  = composition.BandCenter()
df = bc_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[10]:


# miedema
m_feat  = composition.Miedema()
df = m_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[11]:


# stoichiometry
s_feat  = composition.Stoichiometry()
df = s_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[12]:


# t metal fraction
tmf_feat  = composition.TMetalFraction()
df = tmf_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[13]:


# valence orbital
vo_feat  = composition.ValenceOrbital()
df = vo_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[14]:


# yang solid solution
yss_feat  = composition.YangSolidSolution()
df = yss_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[15]:


df.head()


# ## the following three feature-generators also work; however...
# - calculation of cohesive energy takes a long time
# - calculation of atomic packing efficiency takes a  really really long time
# - element fraction adds a lot of null columns. we might not need this

# In[ ]:


# cohesive energy
# long calculation time
ce_feat  = composition.CohesiveEnergy()
df = ce_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[ ]:


# atomic packing efficiency
# very very long calculation time
ape_feat  = composition.AtomicPackingEfficiency()
df = ape_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape


# In[ ]:


# element fraction
# It adds a lot null columns of elements
ef_feat  = composition.ElementFraction()
df = ef_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True) 
df.shape

