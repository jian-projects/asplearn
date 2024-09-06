import random, re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

## Basic LLM template
class ALSC_INS_Base(object):
    def __init__(self, args, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        # self.ret_num = args.model['ret_num'] 

    def score(self, preds, labels, lower=True):
        num_true_positive, num_label, num_pred = 0, 0, 0
        # 每个 pred/truth 包含若干个四元组, 看看预测的四元组是否在真实的四元组中
        for i, (quad_p, quad_l) in enumerate(zip(preds, labels)):
            if lower: quad_p, quad_l = [[it.lower() for it in q] for q in quad_p], [[it.lower() for it in q] for q in quad_l]
            num_pred, num_label = num_pred+len(quad_p), num_label+len(quad_l)
            for quad in quad_p: # 预测中是否存在 真实的四元组？？
                if quad in quad_l: 
                    num_true_positive += 1
            
        precision = float(num_true_positive) / float(num_pred) if num_pred != 0 else 0.0
        recall = float(num_true_positive) / float(num_label) if num_label != 0 else 0.0
        f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
        
        return {
            'p':  round(precision, 4),
            'r':  round(recall, 4),
            'f1': round(f1, 4)
        }

    def get_score(self, preds, labels, stage='valid'):
        return {
            'f1': f1_score(labels, preds, average='macro'),
            'acc': accuracy_score(labels, preds)
        }


        preds_str = np.concatenate(
            [[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rec['outputs']['sequences']] for rec in results]
            )
        labels_str = np.concatenate(
            [[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rec['inputs']['input_ids_t']] for rec in results]
            )
        
        preds_str, labels_str = [p.lower() for p in preds_str], [l.lower() for l in labels_str]
        labels, preds = [eval(l_str) for l_str in labels_str], []
        for p_str in preds_str:
            if not p_str.startswith('['): p_str = p_str[p_str.find("["):p_str.find("]")+1]
            try:
                p = eval(p_str)
            except:
                tmp = [[tmp.strip() for tmp in it.replace("'", "").split(',')] for it in re.findall(r"\((.*?)\)", p_str)]
                p = [(t[0], t[1], t[2], t[3]) if len(t)==4 else () for t in tmp]
            preds.append(p)

        score = self.score(preds, labels)
        return score

    def get_output(self, dataset):
        for stage, samples in dataset.datas.items():
            for s in samples: s['target'] = s['polarity']
        return dataset

    def get_prompt(self, s, trainset, k=1):
        prompt = f"""Please perform Aspect Level Sentiment Classification. Given a sentence and concerned aspect, predict the sentiment polarity of this sentence toward the aspect. Sentiment polarity should be selected from ['negative', 'neutral', 'positive']. Please return the predicted sentiment polarity only, without any other comments or texts. \n"""

        ## 每个类别选 k 个demonstrations
        for lab, idxs in s['retrieval'].items():
            for i in idxs[0:k]:
                prompt += f"\nSentence: {trainset[i]['sentence']}\nAspect: {trainset[i]['aspect']}\nLabel: {trainset[i]['target']}"

        prompt += f"\nInput: {s['sentence']} \nAspect: {s['aspect']}\nLabel: "
        s['prompt'] = prompt

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