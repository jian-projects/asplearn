import numpy as np
from sklearn.metrics import f1_score, accuracy_score

## Basic LLM template
class ALSC_INS_Base(object):
    def __init__(self, args, tokenizer=None) -> None:
        self.args = args
        self.tokenizer = tokenizer
        # self.ret_num = args.model['ret_num'] 

    def get_score(self, preds, labels, stage='valid'):
        return {
            'f1': f1_score(labels, preds, average='macro'),
            'acc': accuracy_score(labels, preds)
        }

    def get_output(self, dataset):
        for stage, samples in dataset.datas.items():
            for s in samples: s['target'] = s['polarity']
        return dataset

    def get_prompt(self, s, trainset, k=1):
        prompt = f"""Please perform the Aspect Level Sentiment Classification task: given a sentence and a specific aspect, predict the sentiment of this sentence toward this aspect. Sentiment must be selected from ['negative', 'neutral', 'positive']. Please return the predicted sentiment only, without any other comments or texts."""

        # ## 每个类别选 k 个demonstrations
        if k > 0: prompt += '\n\nFor example: '
        for lab, idxs in s['retrieval'].items():
            for i in idxs[0:k]:
                prompt += f"\nSentence: {trainset[i]['sentence']}\nAspect: {trainset[i]['aspect']}\nLabel: {trainset[i]['target']}"

        prompt += f"\n\nNow, complete the task: \nSentence: {s['sentence']} \nAspect: {s['aspect']}\nLabel: "

        return prompt
    

## Instruct-ABSA template
class ALSC_INS_2(ALSC_INS_Base):
    def __init__(self, args, dataset) -> None:
        self.args = args
        self.dataset = dataset
    
    def get_output(self, dataset=None):
        if dataset is None: dataset = self.dataset
        for stage, samples in dataset.datas.items():
            for s in samples:

                s['target'] = ', '.join([f"{q[0]}:{q[1]}:{q[2]}:{q[3]}" for q in s['quads']])
        
        return dataset

    def get_prompt(self, s, trainset, stage='train'):
        if not hasattr(self, 'asp_category'): 
            self.asp_category = list(set([q[1] for s in trainset for q in s['quads']]))

        prompt = f"""Definition: The output will be the aspects (both implicit and explicit), the corresponding opinion/describing terms, the aspect category {self.asp_category}, and the sentiment polarity (positive, negative, neutral) of the opinion term. 
        The output should be in the format: aspect:opinion:sentiment:category, aspect:opinion:sentiment:category, ...
        In cases where there are no aspects the output should be NULL:NULL:NULL:NULL."""
        
        idxs = np.arrange(10)
        for i in idxs:
            prompt += f"\nInput: {trainset[i]['sentence']}\nOutput: {trainset[i]['target']}"

            prompt += f"\nInput: {s['sentence']} \nOutput: "
        s['prompt'] = prompt

        return s