# coding=utf-8
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from utils.utils import get_device

from BertBasic.BertBasic import BertBasic
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.utils import InputExample, convert_examples_to_features, convert_features_to_tensors


class TextInferenceModel():
    def __init__(self, weight_path="./parameters/pytorch_model.bin", config_path="./parameters/config.json", vocab_file="./parameters/bert-base-uncased-vocab.txt"):
        self.weight_path = weight_path
        self.config_path = config_path
        self.vocab_file = vocab_file
        self.tokenizer = BertTokenizer.from_pretrained(
        self.vocab_file, do_lower_case=True)
        device, n_gpu = get_device(0)  
        bert_config = BertConfig(self.config_path)
        self.model = BertBasic(bert_config, num_labels=10)
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(device)
        self.model.eval()
        self.label_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.device = device
    def predict(self, text):
        dataloader = self.__load_data(text)
        _, input_ids, input_mask, segment_ids, _ = tuple(dataloader[0])[0]
        with torch.no_grad():
            logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device), labels=None)
            preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        assert(len(outputs)==1)
        return outputs[0]
        
    def __load_data(self, text):
        examples = [InputExample(guid=0, text_a=text, label="1")]
        features = convert_examples_to_features(
            examples, self.label_list, 150, self.tokenizer)

        dataloader = convert_features_to_tensors(features, 1, "test_verbose")


        return dataloader, examples


inference_model = TextInferenceModel()
print(inference_model.predict("While all AMD processors now support 64 bit processing, the Sempron processors are AMD's line, run at somewhat lower clock speeds, and have less L2 cache on chip, making them much less expensive to manufacture while still providing very good performance. (see link 1) The AMD Athlon 64 and 64FX processors (links 2,3)"))
