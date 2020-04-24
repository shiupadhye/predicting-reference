# pronoun-project

### Project
src/
- model.py
- experiment.py

stimuli/
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


### Usage
python run_experiment.py --load_from [datafile] save_to [resultfile] ref_exps ["pron1",.,"name"] prompt_ending [period/connective]

