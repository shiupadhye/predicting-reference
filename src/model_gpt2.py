import re
import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2Tokenizer,GPT2LMHeadModel

class RefExpPredictor(nn.Module):
    """
    Given a stimulus and list of potential referring expressions, computes 
    probability scores for each of the referring expressions. 
    Uses GPT2-XL as base model.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer =GPT2Tokenizer.from_pretrained("gpt2-large")
        self.tokenizer.add_special_tokens({'bos_token':'<|startoftext|>'})
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.softmax = nn.Softmax(dim=0)

    def preprocess(self,text_input):
        bos_token = '<|startoftext|>'
        preprocessed_input = self.tokenizer.bos_token + " " + text_input 
        encoded_input = self.tokenizer.encode(preprocessed_input,add_special_tokens=True)
        return encoded_input

    def compute_probability_scores(self,text_input,ref_exps):
        """
        returns probability scores for referring expressions
        """
        # tokenize
        encoded_input = self.preprocess(text_input)
        print("encoded_input",encoded_input)
        # sanity check
        print("decoded input: ", self.tokenizer.decode(encoded_input))
        predict_after_idx = len(encoded_input)-1
        # sanity check
        print("predict_after: ", self.tokenizer.decode(encoded_input[predict_after_idx]))
        encoded_ref_exps = [self.tokenizer.encode(exp,add_prefix_space=True) for exp in ref_exps]
        # sanity check
        print(ref_exps, encoded_ref_exps)
        encoded_input = torch.tensor(encoded_input).unsqueeze(0)
        # compute probabilities
        probs = []
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded_input)
        logits = output[0]
        prediction_scores = torch.tensor([logits[0,predict_after_idx][ref_id].item() for ref_id in encoded_ref_exps])
        probs = self.softmax(prediction_scores).tolist()
        # sanity checks
        argmax = torch.argmax(logits[0,predict_after_idx]).item()
        print("argmax: ", argmax, "token: ", self.tokenizer.decode(argmax))
        return probs


def main():
    model = RefExpPredictor()
    sent = "Mary aggravated John."
    ref_exps = ["He","She"]
    scores = model.compute_probability_scores(sent,ref_exps)
    print(ref_exps,scores)


if __name__ == "__main__":
    main()

