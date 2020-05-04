"""
usage: python model.py --load_from [datafile] save_to [resultfile] ref_exps ["pron1",.,"name"] prompt_ending [period/connective]
"""
import csv
import argparse
import numpy as np
import model as refexpmodel

#------------------------ Data I/O -------------------------

def load_stimuli(filename,prompt_ending):
    stimuli = []
    with open(filename,'r') as f:
        stimuli = [preprocess_stimuli(line.strip(),prompt_ending) for line in f.readlines()[1:]]
    return stimuli


def save_results(outfile,results,referents):
    headers = ['Stimuli'] + [ref for ref in referents]
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
        ref_exp_probabilities = model.get_probability_scores(stimulus,ref_exps)
        probability_scores = {}
        for i,ref_exp in enumerate(ref_exps):
            probability_scores[ref_exp] = ref_exp_probabilities[i]
        results[stimulus] = probability_scores
    return results  

def run_experiment(datafile,prompt_ending,ref_exps,outfile):
    # load stimuli
    stimuli = load_stimuli(datafile,prompt_ending)
    # init model
    model = refexpmodel.RefExpPredictor()
    results = run_next_word_prediction(model,stimuli,ref_exps)
    save_results(outfile,results,ref_exps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from",type=str,help="path of file containing stimuli",required=True)
    parser.add_argument("--save_to",type=str,help="path of file for storing results",required=True)
    parser.add_argument('--ref_exps',type=str,required=True)
    parser.add_argument('--prompt_ending',type=str,required=True)
    args = vars(parser.parse_args())

    datapath = args['load_from']
    resultpath = args['save_to']
    ref_exps = args['ref_exps']
    ref_exps = [ref for ref in ref_exps.split(",")]
    prompt_ending = args['prompt_ending']

    run_experiment(datapath,prompt_ending,ref_exps,resultpath)

    #datafile = "../stimuli/IC1.csv"
    #prompt_ending = "."
    #outfile = "../results/IC1_exp1A.csv"
    #referents = ["He","She"]
    

if __name__ == "__main__":
    main()
    