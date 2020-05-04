import re
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

    def preprocess(self,text):
        preprocessed_tokens = []
        # specific words to exclude from preprocessing
        exceptions = ["Mr."]
        for token in text.split():
            # check for exceptions
            if token in exceptions:
                split_tokens = [token]
            else:
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
        starter_prompt = """John is a speech pathologist, and he lives in San Francisco with his
        family. Mary is a software engineer, and she also lives in San Francisco with 
        her one year old Golden Retriever."""
        preprocessed_padding_text = self.preprocess(PADDING_TEXT) + [self.tokenizer.eos_token]
        #modified_stimulus = starter_prompt + stimulus
        #preprocessed_stimulus = self.preprocess(modified_stimulus)
        preprocessed_stimulus = self.preprocess(stimulus)
        # encode padding text
        encoded_padding_text = self.tokenizer.convert_tokens_to_ids(preprocessed_padding_text)
        # check prompt ending
        encoded_stimulus = self.tokenizer.convert_tokens_to_ids(preprocessed_stimulus)
        encoded_input = encoded_padding_text + encoded_stimulus
        #print(encoded_input)
        return encoded_input


    def get_probability_scores(self,text_input,ref_exps):
        """
        returns probability scores for referring expressions
        """
        # tokenize
        encoded_input = self.preprocess_and_tokenize(text_input)
        predict_after = len(encoded_input)-1
        # sanity check
        print(self.tokenizer.decode(encoded_input[predict_after]))
        print(self.tokenizer.decode(encoded_input))
        encoded_ref_exps = [self.tokenizer.encode(exp) for exp in ref_exps]
        print(encoded_ref_exps)
        encoded_input = torch.tensor(encoded_input).unsqueeze(0)
        # compute probabilities
        probs = []
        self.model.eval()
        with torch.no_grad():
            output = self.model(encoded_input)
        logits = output[0]
        prediction_scores = torch.tensor([logits[0,predict_after][id].item() for id in encoded_ref_exps])
        probs = self.softmax(prediction_scores).tolist()
        return probs





