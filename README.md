# predicting reference

## Project

### src/
- model_transfoxl.py
- model_gpt2.py
- run_experiment.py

### stimuli/

- IC1
stimuli with subject-biased Implicit Causality verbs (20 files)

- IC2
stimuli with object-biased Implicit Causality verbs (20 files)

- motion
stimuli with motion verbs (18 files)

- transofposs_aspect_imperfective
stimuli with imperfective transfer of possession verbs (18 files)

- transofposs_aspect_perfective
stimuli with perfective transfer of possession verbs (18 files)



### results/

Results of the experiments in .csv format

- transfoxl
	- IC1
		- 20 files
	- IC2
		- 20 files
	- motion
		- 17 files
	- transofposs_aspect_imperfective
		- 18 files
	- transofposs_aspect_perfective
		- 18 files


### reports/
Analysis and Visualization of Results
- Transfoxl-Subject-biased-IC
- Transfoxl-Object-biased-IC
- Transfoxl-Motion
- Transfoxl-transofposs_perfective
- Transfoxl-transofposs_imperfective
- GPT2-Subject-biased-IC
- GPT2-Object-biased-IC
- GPT2-Motion
- GPT2-transofposs_perfective
- GPT2-transofposs_imperfective
- Results


### Usage

For evaluating the models on a single file (recommended for CPU):
```python src/run_experiment.py --model=["transfoxl" or "gpt2"] --load_from=[path-to-input-file] --save_to=[path-to-output-file] --ref_exps=["<pron1>,..,<name1>,.."] --prompt_ending=["<punctuation>" or "<connective>"]```


For evaluating the models on all the files in the directory:
```python src/run_experiment.py --model=["transfoxl" or "gpt2"] --load_from=[path-to-input-directory] --save_to=[path-to-output-directory] --ref_exps=["<pron1>,..,<name1>,.."] --prompt_ending=["<punctuation>" or "<connective>"]```

### Dependencies
- requirements.txt



