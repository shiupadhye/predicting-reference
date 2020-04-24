import torch
import numpy as np
import torch.nn as nn
from transformers import TransfoXLTokenizer,TransfoXLLMHeadModel

class RefExpPredictor(nn.Module):
    """
    Given a stimulus and list of potential referring expressions, computes 
    probability scores for each of the referring expressions. 
    Uses Transformer-XL as base model.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103",eos_token='<eos>')
        self.model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        self.softmax = nn.Softmax(dim=0)

    def preprocess_and_tokenize(self,stimulus):
        PADDING_TEXT = "In 1991, the remains of Russian Tsar Nicholas II and his family \
        (except for Alexei and Maria) are discovered. \
        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the \
        remainder of the story. 1883 Western Siberia,\
        a young Grigori Rasputin is asked by his father and a group of men to perform magic. \
        Rasputin has a vision and denounces one of the men as a horse thief. Although his \
        father initially slaps him for making such an accusation, Rasputin watches as the \
        man is chased outside and beaten. Twenty years later, Rasputin sees a vision of \
        the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, \
        with people, even a bishop, begging for his blessing."
        encoded_prompt = self.tokenizer.encode(PADDING_TEXT,add_space_before_punct_symbol=True) + [self.tokenizer.eos_token_id]
        if stimulus.endswith("."):
            encoded_stimulus= self.tokenizer.encode(stimulus,add_space_before_punct_symbol=True) + [self.tokenizer.eos_token_id]
        else:
            encoded_stimulus= self.tokenizer.encode(stimulus,add_space_before_punct_symbol=True)
        encoded_input = encoded_prompt + encoded_stimulus
        return encoded_input


    def get_probability_scores(self,text_input,ref_exps):
        """
        returns probability scores for referring expressions
        """
        # tokenizer
        encoded_input = self.preprocess_and_tokenize(text_input)
        predict_after = len(encoded_input)-1
        # sanity check
        #print(self.tokenizer.decode(encoded_input[predict_after]))
        #print(encoded_input)
        #print(self.tokenizer.decode(encoded_input))
        encoded_ref_exps = [self.tokenizer.encode(exp,add_space_before_punct_symbol=True) for exp in ref_exps]
        encoded_input = torch.tensor(encoded_input).unsqueeze(0)
        # compute probabilities
        probs = []
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded_input)
        logits = output[0]
        prediction_scores = self.softmax(logits[0,predict_after])
        probs = [prediction_scores[id[0]].item() for id in encoded_ref_exps]
        return probs





