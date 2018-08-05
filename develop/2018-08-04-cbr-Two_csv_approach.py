
# coding: utf-8

# In[124]:


import pandas as pd
from os import environ
from citrination_client import CitrinationClient
from citrination_client import *
from pypif import pif
import csv

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction


# In[128]:


client = CitrinationClient(environ['CITRINATION_API_KEY'], 'https://citrination.com')
dataset_id = '151803'


# ### Create csv file with properties and conditions

# In[182]:


# This function creates a csv files with columns for:
# - the property specified in prop_name
# - its units
# - a corresponding property (e.g. temperature)
# - crystallinity
# - a refence

def create_csv_prop(prop_name, cond_name):
    value_query = FieldQuery(extract_as=prop_name, extract_all=True)
    property_query = PropertyQuery(name=FieldQuery(filter=[Filter(equal=prop_name)]), value=value_query)
    formula_query = ChemicalFieldQuery(extract_as="formula") # Defines how to extract formula(?)
    system_query = PifSystemQuery(chemical_formula=formula_query, properties=property_query)
    dataset_query = DatasetQuery(id=[Filter(equal=dataset_id)]) # selects dataset
    
    data_query = DataQuery(dataset=dataset_query, system=system_query)
    pif_query = PifSystemReturningQuery(size=5000, random_results=True, query=data_query)

    search_result = client.search.pif_search(pif_query) # Actual pif query against the citrination database

    print("We found {} records".format(len(search_result.hits)))
    #print([x.extracted for x in search_result.hits[0:2]])
    
    
    rows = []
    pif_records = [x.system for x in search_result.hits]
    for system in pif_records:
        if "x" not in system.chemical_formula and "." not in system.chemical_formula:
            crystallinity_val = 'Unknown'
            for prop in system.properties:
                
                if prop.name == 'Crystallinity': # look up crystallinity value for this material
                    crystallinity_val = prop.scalars[0].value
                    
                if prop.name == prop_name: #only materials with the desired property
                    

                    for cond in prop.conditions: # loop through all conditions associated with the desired property

                        if cond_name == 'None': # If no condition is specified (use all materials)
                            row = [system.chemical_formula, prop.scalars[0].value, prop.units,  crystallinity_val,                                   system.references[0].citation]
                            rows.append(row)
                        else: # If a condition is specified
                            if cond.name == cond_name:
                                if len(prop.scalars) == len(cond.scalars):

                                    for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
                                        row = [system.chemical_formula, prop_sca.value, prop.units, cond_sca.value,                                                crystallinity_val, system.references[0].citation]
                                        rows.append(row)

                            

                                


    with open(prop_name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        if cond_name == 'None':
            writer.writerow(['Formula', prop_name, 'Unit', 'Crystallinity', 'Reference'])
        else:
            writer.writerow(['Formula', prop_name, 'Unit', cond_name, 'Crystallinity', 'Reference'])
            
        writer.writerows(rows)


# In[183]:


condition_names = ['Specimen Number', 'Diagonal Direction', 'Typical Range', 'Test Type', 'Annealing Time at 1600 $^{\\circ}$C', 'x of 14Ni:xCof Additive', 'Film Thickness', 'Test Condition', 'Mass Fraction of Additive', 'Environment', 'Number of Quenches', 'Porosity', 'Test Temperature', 'Indentation Load', 'Density', 'Loadin Rate', 'Specimen Code', 'Deflection Angle', 'Method', 'Volume Fraction of SiC', 'Phase', 'Sintering Condition', 'Volume Fraction of BA', 'H/ERatio', 'Number of Tests', 'Specimen Thickness', 'Quenching Temperature', 'Numbers of Indents', 'Volume Fraction of TiO2', 'Additive', 'Test Environment', 'x of Alx', 'Specimen Condition', 'Crack Type', 'Notch Width', 'Notch Polishing', 'Orientation', 'Tensile Surface', 'Hot Pressing Time', 'Temperature', 'Relative Density', 'Volume Fraction of HA', 'Volume Fraction of TCP', 'Crystal Plane', 'Batch Number', 'Sample Number', 'Procedure', 'Sintering Temperature', 'Crosshead Speed', 'Mass Fraction of SiC', 'Load', 'Mole Fraction of Additive', 'Method Name', 'Tensile Axis', 'Material Type', 'Volume Swelling', 'Init. Relative Density', 'Measurement Temperature', 'Crack Length', 'Devitrified', 'Type of Zone', 'Crystallizing Treatment', 'Specimen Size', 'Mass Fraction of Ni', 'Fracture Plane', 'Volume Fraction of Porosity', 'Grain Size']
#sorted(condition_names)


# ### Featurizing with matminer

# In[184]:


def featurize_csv(file_name, prop_name):
    
    # Read csv file
    df = pd.read_csv(file_name)
    print('DataFrame shape pre-grouping: ', df.shape)
    
    # Remove errors given in parentheses from numerical columns
    import re
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
    df[prop_name] = df[prop_name].transform(remove_par_errs).astype(float)
    
    # Group samples by chemical formula
    df = df.groupby(['Formula','Crystallinity'], as_index=False).mean()
    print('DataFrame shape post-grouping: ', df.shape)    
    
    # Create composition column
    df["composition"] = df["Formula"].transform(str_to_composition)
    
    #Featurize dataframe
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    
    # Write new csv file
    new_file_name =  file_name[:-4] + '_featurized.csv'
    df.to_csv(new_file_name)


# ### Execute functions

# In[186]:


prop_name = 'Fracture Toughness'
file_name = prop_name + '.csv'
create_csv_prop(prop_name, 'Temperature')
featurize_csv(file_name, prop_name)


# ### Test of featurization (used to develop above function)

# In[172]:


df = pd.read_csv('Density.csv')


# In[173]:


df.head()


# In[176]:


# Remove errors given in parentheses from numerical columns
import re
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

df['Density'] = df['Density'].transform(remove_par_errs).astype(float)

#conv_to_float(df, 'Axis Length').dropna().head()


# In[177]:


df.shape


# In[179]:


df = df.groupby(['Formula','Crystallinity'], as_index=False).mean()


# In[180]:


df.shape


# In[112]:


df["composition"] = df["Formula"].transform(str_to_composition)


# In[113]:


ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)


# In[114]:


df.head()


# In[107]:


df.to_csv('fracture_toughness_featurized.csv')

