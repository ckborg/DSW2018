
# coding: utf-8

# In[103]:


import pandas as pd
from os import environ
from citrination_client import CitrinationClient
from citrination_client import *
from pypif import pif
import csv

from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction


# In[7]:


client = CitrinationClient(environ['CITRINATION_API_KEY'], 'https://citrination.com')
dataset_id = '151803'


# ### Create csv file with properties and conditions

# In[78]:


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

    search_result = client.search.pif_search(pif_query)

    print("We found {} records".format(len(search_result.hits)))
    print([x.extracted for x in search_result.hits[0:2]])
    
    

    #condition_names = []
    rows = []
    pif_records = [x.system for x in search_result.hits]
    for system in pif_records:
        if "x" not in system.chemical_formula and "." not in system.chemical_formula:
            crystallinity_val = 'Unknown'
            for prop in system.properties:
                
                if prop.name == 'Crystallinity':
                    crystallinity_val = prop.scalars[0].value
                if prop.name == prop_name: #only entries with the desired property
                    

                    for cond in prop.conditions:
                        #condition_names.append(cond.name)

                        if cond.name == cond_name:
                            if len(prop.scalars) == len(cond.scalars):

                                for prop_sca, cond_sca in zip(prop.scalars, cond.scalars):
                                    row = [system.chemical_formula, prop_sca.value, prop.units, cond_sca.value,  crystallinity_val, system.references[0].citation]
                                    rows.append(row)

                                


    with open('fracture_toughness.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Formula', prop_name, 'Unit', cond_name, 'Crystallinity', 'Reference'])
        writer.writerows(rows)
    #print(set(condition_names))


# In[77]:


create_csv_prop('Fracture Toughness', 'Temperature')


# In[60]:


condition_names = ['Specimen Number', 'Diagonal Direction', 'Typical Range', 'Test Type', 'Annealing Time at 1600 $^{\\circ}$C', 'x of 14Ni:xCof Additive', 'Film Thickness', 'Test Condition', 'Mass Fraction of Additive', 'Environment', 'Number of Quenches', 'Porosity', 'Test Temperature', 'Indentation Load', 'Density', 'Loadin Rate', 'Specimen Code', 'Deflection Angle', 'Method', 'Volume Fraction of SiC', 'Phase', 'Sintering Condition', 'Volume Fraction of BA', 'H/ERatio', 'Number of Tests', 'Specimen Thickness', 'Quenching Temperature', 'Numbers of Indents', 'Volume Fraction of TiO2', 'Additive', 'Test Environment', 'x of Alx', 'Specimen Condition', 'Crack Type', 'Notch Width', 'Notch Polishing', 'Orientation', 'Tensile Surface', 'Hot Pressing Time', 'Temperature', 'Relative Density', 'Volume Fraction of HA', 'Volume Fraction of TCP', 'Crystal Plane', 'Batch Number', 'Sample Number', 'Procedure', 'Sintering Temperature', 'Crosshead Speed', 'Mass Fraction of SiC', 'Load', 'Mole Fraction of Additive', 'Method Name', 'Tensile Axis', 'Material Type', 'Volume Swelling', 'Init. Relative Density', 'Measurement Temperature', 'Crack Length', 'Devitrified', 'Type of Zone', 'Crystallizing Treatment', 'Specimen Size', 'Mass Fraction of Ni', 'Fracture Plane', 'Volume Fraction of Porosity', 'Grain Size']
sorted(condition_names)


# ### Featurizing with matminer

# In[108]:


df = pd.read_csv('fracture_toughness.csv')


# In[88]:


df.head()


# In[110]:


df = df.groupby(['Formula','Crystallinity'], as_index=False).mean()


# In[112]:


df["composition"] = df["Formula"].transform(str_to_composition)


# In[113]:


ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)


# In[114]:


df.head()


# In[107]:


df.to_csv('fracture_toughness_featurized.csv')

