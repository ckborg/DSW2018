
# coding: utf-8

# In[1]:


import pandas as pd
import re

from matminer.utils.conversions import str_to_composition
from matminer.featurizers import composition


# ## Create Data Set

# In[2]:


df  = pd.read_pickle('../deliver/NIST_CeramicDataSet.pkl')


# ## Composition parser via regular expression

# In[3]:


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


# ##  Manually Clean up Data Set

# In[4]:


# problematic entries
print('1185: ' + df.loc[1185,'Density'])
print('2673: ' + df.loc[2673, 'Fracture Toughness'])
print('494: ' + df.loc[494, "chemicalFormula"])
print('1007: ' + df.loc[1007,"chemicalFormula"])
print('2696: ' + df.loc[2696, "chemicalFormula"])
print('2998: ' + df.loc[2998,"chemicalFormula"])
print('3892: ' + df.loc[3892,"chemicalFormula"])
print('3863: ' + df.loc[3863, "chemicalFormula"])


# In[5]:


# update chemicalFormula with cleanUp function
df["chemicalFormula"] = df["chemicalFormula"].map(cleanUp)
# update chemicalFormula by manually clean-up
df.loc[1185,"Density"] = '3.16'
df.loc[2673, 'Fracture Toughness'] = '3.1'
df.loc[494, "chemicalFormula"] = 'Zr0.5Y2O3'
df.loc[1007,"chemicalFormula"] = 'TiC'
df.loc[2696,"chemicalFormula"] = 'Si3N4.xBaOY2O3'
df.loc[3892,"chemicalFormula"] = 'BN'
df.loc[3863,"chemicalFormula"] = 'C'
df.loc[2998,"chemicalFormula"] = '3Al2O3.2SiO2.Si2–xAlxO1+xN2–x'


# ## sub-DataFrame with feature of Fracture Toughness

# In[6]:


# create sub-dataset with valid Fracture Toughness feature
df_ft = df[df.isnull()['Fracture Toughness'] == False]


# In[7]:


# sub-dataset only focus on certain columns
df_ft = df_ft[['chemicalFormula','names', 'preparation','Chemical Family','Crystallinity','Fracture Toughness']]
# make all features are string type for doing groupby later
df_ft = df_ft.astype(str)
df_ft = df_ft.fillna('0')
# convert Fracture Toughness from string to float
df_ft['Fracture Toughness'] = df_ft['Fracture Toughness'].astype(float)


# In[8]:


df_ft = df_ft.groupby( ['chemicalFormula','names', 'preparation','Chemical Family','Crystallinity'])['Fracture Toughness'].mean()
df_ft = pd.DataFrame(df_ft).reset_index()
#print(df2.loc[3,'chemicalFormula'])
#df_ft.to_csv('Fracture-Toughness-1.csv')


# In[9]:


# create composition feature based on parsed formula
df_ft["flatFormula"] = df_ft["chemicalFormula"].map(formula_parser)
df_ft["composition"] =df_ft["flatFormula"].transform(str_to_composition)
df_ft.shape


# ### add features based on composition (composition with floats)

# In[10]:


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
# valence orbital
vo_feat  = composition.ValenceOrbital()
df_ft = vo_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# yang solid solution
yss_feat  = composition.YangSolidSolution()
df_ft = yss_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 

df_ft.shape


# In[11]:


# cohesive energy
ce_feat  = composition.CohesiveEnergy()
df_ft = ce_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 
# atomic packing efficiency
ape_feat  = composition.AtomicPackingEfficiency()
df_ft= ape_feat.featurize_dataframe(df_ft, col_id="composition", ignore_errors=True) 

df_ft.shape


# ## save data

# In[12]:


# save Fracture Toughness sub-dataset as csv
df_ft.to_csv('fracture_toughness_clean.csv')

