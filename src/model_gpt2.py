import re
import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2Tokenizer,GPT2LMHeadModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RefExpPredictor(nn.Module):
    """
    Given a stimulus and list of potential referring expressions, computes 
    probability scores for each of the referring expressions. 
    Uses Hugginface's GPT2-large implementation as base model 
    https://huggingface.co/transformers/model_doc/gpt2.html

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
        encoded_input = self.preprocess(text_input)
        predict_after_idx = len(encoded_input)-1
        encoded_ref_exps = [self.tokenizer.encode(exp,add_prefix_space=True) for exp in ref_exps]
        encoded_input = torch.tensor(encoded_input).unsqueeze(0).to(device)
        probs = []
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded_input)
        logits = output[0]
        prediction_scores = torch.tensor([logits[0,predict_after_idx][ref_id].item() for ref_id in encoded_ref_exps])
        probs = self.softmax(prediction_scores).tolist()
        return probs


