"""
usage: python model.py --load_from [datafile] save_to [resultfile] ref_exps ["pron1",.,"name"] prompt_ending [period/connective]
"""
import os
import csv
import argparse
import torch
import numpy as np
import model_transfoxl as refexpmodel_transfoxl
import model_gpt2 as refexpmodel_gpt2

refexpmodels = {'transfoxl':refexpmodel_transfoxl,'gpt2':refexpmodel_gpt2}

#------------------------ Data I/O -------------------------

def load_stimuli(filename,prompt_ending):
    stimuli = []
    with open(filename,'r') as f:
        stimuli = [preprocess_stimuli(line.strip(),prompt_ending) for line in f.readlines()[1:]]
    return stimuli


def save_results(outfile,results,referents):
    headers = ['stimulus'] + [ref for ref in referents]
    with open(outfile,'w') as csvfile:
        writer = csv.writer(csvfile,delimiter=",")
        writer.writerow(headers)
        for stimuli, probability_scores in results.items():
            ref_probs = [prob for prob in list(probability_scores.values())]
            row = [stimuli] + ref_probs
            writer.writerow(row)

# ------------------- Preprocessing --------------------------

def preprocess_stimuli(stimulus,prompt_ending):
    if prompt_ending == ".":
        stimulus = stimulus + prompt_ending
    else:
        stimulus = stimulus + " " + prompt_ending
    return stimulus

#------------------------ Experiment -------------------------

def run_next_word_prediction(model,stimuli,ref_exps):
    results = {}
    for stimulus in stimuli:
        ref_exp_probabilities = model.compute_probability_scores(stimulus,ref_exps)
        probability_scores = {}
        for i,ref_exp in enumerate(ref_exps):
            probability_scores[ref_exp] = ref_exp_probabilities[i]
        results[stimulus] = probability_scores
    return results  

def run_experiment(model_type,device,datafile,outfile,ref_exps,prompt_ending):
    # determine model
    m = refexpmodels[model_type]
    print("running experiments using model: %s" % model_type)
    # load stimuli
    stimuli = load_stimuli(datafile,prompt_ending)
    # init model
    model = m.RefExpPredictor()
    model.to(device)
    results = run_next_word_prediction(model,stimuli,ref_exps)
    save_results(outfile,results,ref_exps)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,help="model",required=True)
    parser.add_argument("--load_from",type=str,help="path of dir that contains stimuli",required=True)
    parser.add_argument("--save_to",type=str,help="path of dir for storing results",required=True)
    parser.add_argument('--ref_exps',type=str,required=True)
    parser.add_argument('--prompt_ending',type=str,required=True)
    args = vars(parser.parse_args())


    model_type = args['model']
    input_dir = args['load_from']
    output_dir = args['save_to']
    ref_exps = args['ref_exps']
    ref_exps = [ref for ref in ref_exps.split(",")]
    prompt_ending = args['prompt_ending']

    
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_file = os.path.join(input_dir,file)
            output_file = os.path.join(output_dir,file)
            run_experiment(model_type,device,input_file,output_file,ref_exps,prompt_ending)


    #run_experiment(model_type,device,input_dir,output_dir,ref_exps,prompt_ending)

    

if __name__ == "__main__":
    main()
    