# DSW2018

## Description
Investigation of a structural ceramics database (WebSCD) curated by the National Institute of Standards and Technology (NIST).
3-week project as part of the UC Berkeley Data Science workshop (DSW2018).
[WebSCD]("https://srdata.nist.gov/CeramicDataPortal/scd")


## Results at-a-glance

~4000 material entries
Matminer was used to generate features based on composition.


Most represented mechanical properties: Elastic modulus, flexural strength, fracture toughness.

![plot of most represented propeties](/images/properties.png)

Data distribution for properties of interest:

* Flexural strength

![distribution](/images/flexural_strength.png)

* Fracture toughness

![distribution](/images/fracture_toughness.png)

* Elastic modulus

![distribution](/images/elastic_modulus.png)



Random forest regression:

* Flexural strength

![prediction vs. actual curve](/images/flexural_rf.png)

* Fracture toughness

![prediction vs. actual curve](/images/fracture_rf.png)

* Elastic modulus

![prediction vs. actual curve](/images/elastic_rf.png)


Important features:

* Flexural strength

![Important features](/images/flexural_features.png)

* Fracture toughness

![Important features](/images/fracture_features.png)

* Elastic modulus

![Important features](/images/elastic_features.png)


## Team members
[Prabudhya Bhattacharyya]("http://physics.berkeley.edu/people/graduate-student/prabudhya-bhattacharyya")

[Chris Bronner]("https://chrisbronner.com/")

[Steve Drapcho]("https://www.linkedin.com/in/steven-drapcho-2b939b149/")

[Chih-Hao Hsu]("http://cedrichsu.com/")





