import csv
import model as refexpmodel
import numpy as np

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

# -------------------- Postprocessing ------------------------
def normalize(probability_scores):
    denom = np.sum(np.array(list(probability_scores.values())))
    for ref, score in probability_scores.items():
        probability_scores[ref] = score/denom
    return probability_scores

def normalize_results(results):
    normalized_results = {}
    for prompt, probability_scores in results.items():
        normalized_probability_scores = normalize(probability_scores)
        normalized_results[prompt] = normalized_probability_scores
    return normalized_results


#------------------------ Experiment -------------------------

def run_next_word_prediction(model,stimuli,referents):
    results = {}
    for stimulus in stimuli:
        referent_probabilities = model.predict(stimulus,referents)
        probability_scores = {}
        for i,referent in enumerate(referents):
            probability_scores[referent] = referent_probabilities[i]
        results[stimulus] = probability_scores
        normalized_results = normalize_results(results)
    return results  

def run_experiment(datafile,prompt_ending,referents,outfile):
    # load stimuli
    stimuli = load_stimuli(datafile,prompt_ending)
    # init model
    model = refexpmodel.RefExpPredictor()
    results = run_next_word_prediction(model,stimuli,referents)
    save_results(outfile,results,referents)


def main():
    #datafile = "../stimuli/IC2.csv"
    #prompt_ending = "because"
    #outfile = "../results/IC2_exp1A_2-x.csv"
    #referents = ["he","she"]
    #run_experiment(datafile,prompt_ending,referents,outfile)

if __name__ == "__main__":
    main()
    