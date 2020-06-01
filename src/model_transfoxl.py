import re
import torch
import numpy as np
import torch.nn as nn
from transformers import TransfoXLTokenizer,TransfoXLLMHeadModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RefExpPredictor(nn.Module):
    """
    Given a stimulus and list of potential referring expressions, computes 
    probability scores for each of the referring expressions. 
    Uses Hugginface's Transformer-XL implementation as base model 
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103",eos_token='<eos>')
        self.tokenizer.add_special_tokens({'bos_token':'<sos>'})
        self.model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        self.softmax = nn.Softmax(dim=0)

    def preprocess(self,text):
        preprocessed_tokens = []
        for token in text.split():
            split_tokens = re.findall(r"[\w]+|[.,!?;]",token)
            if len(split_tokens) == 1:
                preprocessed_tokens.append(split_tokens[0])
            else:
                for token in split_tokens:
                    preprocessed_tokens.append(token)
        return preprocessed_tokens


    def preprocess_and_tokenize(self,stimulus):
        PADDING_TEXT =  """In 1991, the remains of Russian Tsar Nicholas II and his family
        (except for Alexei and Maria) are discovered. The voice of Nicholas's young son, 
        Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western 
        Siberia, a young Grigori Rasputin is asked by his father and a group of men to 
        perform magic. Rasputin has a vision and denounces one of the men as a horse thief. 
        Although his father initially slaps him for making such an accusation, Rasputin 
        watches as the man is chased outside and beaten. Twenty years later, Rasputin 
        sees a vision of the Virgin Mary, prompting him to become a priest. Rasputin 
        quickly becomes famous, with people, even a bishop, begging for his blessing."""
        #modified_padding_text = self.tokenizer.bos_token + PADDING_TEXT
        preprocessed_padding_text = [self.tokenizer.bos_token] + self.preprocess(PADDING_TEXT) + [self.tokenizer.eos_token]
        #modified_stimulus = starter_prompt + stimulus
        preprocessed_stimulus = [self.tokenizer.bos_token] + self.preprocess(stimulus)
        # encode padding text
        encoded_padding_text = self.tokenizer.convert_tokens_to_ids(preprocessed_padding_text)
        # check prompt ending
        encoded_stimulus = self.tokenizer.convert_tokens_to_ids(preprocessed_stimulus)
        encoded_input = encoded_padding_text + encoded_stimulus
        return encoded_input


    def compute_probability_scores(self,text_input,ref_exps):
        """
        returns probability scores for referring expressions
        """
        # tokenize
        encoded_input = self.preprocess_and_tokenize(text_input)
        predict_after_idx = len(encoded_input)-1
        # sanity check
        encoded_ref_exps = [self.tokenizer.encode(exp) for exp in ref_exps]
        # sanity check
        encoded_input = torch.tensor(encoded_input).unsqueeze(0).to(device)
        # compute probabilities
        probs = []
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded_input)
        logits = output[0]
        prediction_scores = torch.tensor([logits[0,predict_after_idx][ref_id].item() for ref_id in encoded_ref_exps])
        probs = self.softmax(prediction_scores).tolist()
        return probs



