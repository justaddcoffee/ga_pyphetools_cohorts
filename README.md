# Genetic algorithm 
This repo is an implementation of a genetic algorithm to evolve "profiles" (sets of
HPO terms) that can be used to predict the presence of a disease, given a set of phenotypic features (also a set of HPO terms) for a new patient.

Basically we start with a random set of profiles, and then we evaluate them using
labeled data (patients with and without the disease). Then we select the best n profiles
using "termset semantic similarity" also known as Phenomizer score (basically the 
average semantic similarity between the best matches between HPO terms in the profile 
and the HPO terms for patients with the disease). 

We then take and mutate them (cross-over, add/delete terms, and mutate weights for each 
term in the profile) to create a new generation of profiles. We repeat this process for 
a number of generations.


## Installation

```shell
git clone https://github.com/justaddcoffee/ga_pyphetools_cohorts.git
cd ga_pyphetools_cohorts
pip install poetry # if not already installed
poetry install
```

## Usages

### Make phenopackets for a synthetic cohort
```shell
ga make-synthetic-phenopackets -o data/synthetic_phenopackets
```

### Run the genetic algorithm
```shell
ga run -p /path/to/your/phenopackets/dir/ -d "disease of interest"
```
```shell
# for example
ga run -p data/synthetic_phenopackets -d "Marfan syndrome"
```

