# pronoun-project

### Project
src/
- model.py
- run_experiment.py

stimuli/v1/
Contains stimuli with "John" and "Mary" as the subject/object pairs

- IC1.csv 
stimuli with subject-biased Implicit Causality verbs

- IC2.csv 
stimuli with object-biased Implicit Causality verbs

- motion.csv 
stimuli with motion verbs

- transofposs.csv
stimuli with transfer of possession verbs 

- aspect_imperfective.csv
stimuli with imperfective transfer of possession verbs

- aspect_perfective.csv
stimuli with perfective transfer of possession verbs

stimuli/v2/
Contains stimuli with "Man" and "Woman" as the subject/object pairs

- IC1.csv 
stimuli with subject-biased Implicit Causality verbs

- IC2.csv 
stimuli with object-biased Implicit Causality verbs

- motion.csv 
stimuli with motion verbs

- transofposs.csv
stimuli with transfer of possession verbs 

- aspect_imperfective.csv
stimuli with imperfective transfer of possession verbs

- aspect_perfective.csv
stimuli with perfective transfer of possession verbs


results/
Results of the experiments in .csv format

notebooks/
Visualization / analysis of results 


### Usage
```python run_experiment.py --load_from=[path] --save_to=[path] --ref_exps=["<pron1>,..,<name1>,.."] --prompt_ending=["<punctuation>" or "<connective>"]```


