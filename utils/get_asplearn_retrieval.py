
import torch
import torch.nn.functional as F
import numpy as np  

def get_retrieval(args, dataset, bank):
    if args.llm['retrieve'][1] == 'asp':
        mark = 'aspect'
        print("Error: Retrieve Type is Aspect !!!")
    elif args.llm['retrieve'][1] == 'sent':
        mark = 'sentence'
        print("Error: Retrieve Type is Sentence !!!")
    else: print("Error: Retrieve Type is None !!!")

    train_labels = [dataset.tokenizer_["labels"]["l2i"][s["polarity"]] for s in dataset.datas['train']]
    if bank is not None:
        embedding_train = torch.stack(bank['train'][mark])
        if 'prob' in bank['train']: probs_train = torch.stack(bank['train']['prob'])
        embedding_test = torch.stack(bank['test'][mark])
    for s in dataset.datas['test']:
        if bank is not None: 
            sim = F.cosine_similarity(embedding_test[s['index']], embedding_train, dim=-1)
        else: sim = torch.rand(len(dataset.datas['train'])) # 随机检索
        s['retrieval'] = {}
        for lab in range(dataset.num_classes):
            sim_order = torch.argsort(sim, descending=True)
            sim_lab_order = [idx.item() for idx in sim_order if train_labels[idx] == lab]
            if args.llm['retrieve'][0] == 'asplearn':
                tmp = sim_lab_order[0:20]
                probs = np.array([probs_train[t][train_labels[t]] for t in tmp])
                norm_probs = probs / probs.sum()
                s['retrieval'][lab] = [tmp[i] for i in np.sort(np.random.choice(len(tmp), 10, p=norm_probs, replace=False))]
            else:
                s['retrieval'][lab] = sim_lab_order[0:10]

            # s['retrieval'][lab] = sim_lab_order[0:10] # 性能差的要死 ……
            # s['retrieval'][lab] = sim_lab_order[-10:] # 更差了啊
            
            # tmp = []
            # for _ in range(5):
            #     ## 1. 首 中 尾
            #     tmp.extend([sim_lab_order.pop(0), sim_lab_order.pop(len(sim_lab_order)//2), sim_lab_order.pop(-1)])
            # s['retrieval'][lab] = tmp

            # s['retrieval'][lab] = np.concatenate([[sim_lab_order[i], sim_lab_order[-i-1]] for i in range(5)]) # sim_lab_order[0:10]


def get_retrieval_(args, dataset, bank):
    """
    aspect 粗选， sentence 精选
    """

    train_labels = [dataset.tokenizer_["labels"]["l2i"][s["polarity"]] for s in dataset.datas['train']]
    if bank is not None:
        train_a = torch.tensor(bank['train']['aspect']) if args.llm['retrieve'][0] == 'sbert' else torch.stack(bank['train']['aspect'])
        test_a = torch.tensor(bank['test']['aspect']) if args.llm['retrieve'][0] == 'sbert' else torch.stack(bank['test']['aspect'])
        train_s = torch.tensor(bank['train']['sentence']) if args.llm['retrieve'][0] == 'sbert' else torch.stack(bank['train']['sentence'])
        test_s = torch.tensor(bank['test']['sentence']) if args.llm['retrieve'][0] == 'sbert' else torch.stack(bank['test']['sentence'])
    for s in dataset.datas['test']:
        if bank is not None: 
            sim_a = F.cosine_similarity(test_a[s['index']], train_a, dim=-1)
        else: sim_a = torch.rand(len(dataset.datas['train'])) # 随机检索
        s['retrieval'] = {}
        for lab in range(dataset.num_classes):
            sim_order = torch.argsort(sim_a, descending=True)
            sim_lab_order = [idx.item() for idx in sim_order if train_labels[idx] == lab]
            if bank is None: 
                s['retrieval'][lab] = sim_lab_order[0:10]
            else:
                temp = sim_lab_order[0:20]
                sim_s = F.cosine_similarity(test_s[s['index']], train_s[temp], dim=-1)
                sim_s_order = torch.argsort(sim_s, descending=False)
                s['retrieval'][lab] = [temp[i] for i in sim_s_order[0:10]]