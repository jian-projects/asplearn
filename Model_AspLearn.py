
import torch, os, copy, random, sys
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

from transformers import logging
logging.set_verbosity_error()

# config 中已经添加路径了
from utils_processor import *
from utils_model import *
from data_loader import ALSCDataModule


class ALSCDataModule_(ALSCDataModule):
    def setup_(self, tokenizer, stage=None):
        self.tokenizer = tokenizer
        for stage, samples in self.datas.items():
            if samples is None: continue
            self.info['class_category'][stage] = {l: 0 for l in self.tokenizer_['labels']['i2l'].keys()}
            for sample in samples:
                if 'sentence' not in sample: sample['sentence'] = ' '.join(sample['tokens'])
                embedding_sent = tokenizer.encode(sample['sentence'], return_tensors='pt')[0]
                embedding_asp = tokenizer.encode(sample['aspect'], return_tensors='pt')[0][1:]
                sample['input_ids'] = torch.cat([embedding_sent, embedding_asp])
                sample['attention_mask'] = torch.ones_like(sample['input_ids'])
                sample['token_type_ids'] = torch.cat([torch.zeros_like(embedding_sent), torch.ones_like(embedding_asp)])
                sample['label'] = self.tokenizer_['labels']['l2i'][sample['polarity']]
                sample['stage'] = stage
                
                # ############################################################################
                # new_tokens, input_ids, token_type_ids = [], [], []
                # for ti, token in enumerate(sample['tokens']):
                #     if ti == sample['aspect_pos'][0]: 
                #         new_tokens.extend(['[', token])
                #         iids = tokenizer.encode(f"[ {token}")[1:-1]
                #         input_ids.extend(iids)
                #         token_type_ids.extend([1]*len(iids))
                #     elif ti == sample['aspect_pos'][1]:
                #         new_tokens.extend([']', token])
                #         iids = tokenizer.encode(f"] {token}")[1:-1]
                #         input_ids.extend(iids)
                #         token_type_ids.extend([1]*len(iids))
                #     else: 
                #         new_tokens.append(token)
                #         iids = tokenizer.encode(token)[1:-1]
                #         input_ids.extend(iids)
                #         token_type_ids.extend([0]*len(iids))
                # sample['sentence'] = ' '.join(new_tokens)
                # sample['input_ids'] = torch.tensor([tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id])
                # sample['token_type_ids'] = torch.tensor([0] + token_type_ids + [0])
                # sample['attention_mask'] = torch.ones_like(sample['input_ids'])
                # ############################################################################

                self.info['class_category'][stage][sample['label']] += 1

    def collate_fn(self, samples):
        # if samples[0]['stage'] == 'train':
        #     samples = self.setup(samples) # aspect 加噪

        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs

def config_for_model(args, scale='base'):
    scale = args.model['scale'] if 'scale' in args.model else scale
    if args.model['arch'] == 'glove':
        args.model['plm'] = args.file['plm_dir'] + 'glove/'
    if args.model['arch'] == 'bert':
        if scale == 'base':
            args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}-uncased'
        else: args.model['plm'] = args.file['plm_dir'] + f'bert-{scale}'
    if args.model['arch'] == 'deberta':
        args.model['plm'] = args.file['plm_dir'] + f'deberta-{scale}'
    if args.model['arch'] == 'roberta':
        args.model['plm'] = args.file['plm_dir'] + f'roberta-{scale}'
        
    args.model['save_path'] = f"{args.file['save_dir']}{args.train['tasks'][-1]}-{args.model['arch']}-{args.model['scale']}.pt"
    sbert = "all-roberta-large-v1" if scale == 'large' else 'all-distilroberta-v1'
    args.model['sbert'] = f"{args.file['plm_dir']}/sbert/{sbert}" # 
    args.model['data_dir'] = f"{args.file['cache_dir']}{args.train['tasks'][-1]}/"
    if not os.path.exists(args.model['data_dir']): os.makedirs(args.model['data_dir']) # 创建路径
    args.model['data'] = args.file['data_dir']+f"{args.train['tasks'][-1]}/{args.model['name']}_for_all.pt"

    # args.model['optim_sched'] = ['AdamW', 'linear']
    # args.model['optim_sched'] = ['AdamW', 'cosine']
    # if args.model['arch'] == 'glove':
    #     args.model['optim_sched'] = ['AdamW', 'linear']
    
    args.model['epoch_every'] = False # 每个 epoch 前处理
    args.model['epoch_before'] = False # epoch 前处理

    args.model['store_first'] = True
    return args

def import_model(args):
    ## 1. 模型参数配置
    args = config_for_model(args) # 添加模型参数, 获取任务数据集

    ## 2. 数据集配置
    dataset = torch.load(args.model['data'])
    # if os.path.exists(data_path):
    #     dataset = torch.load(data_path)
    # else:
    #     data_dir = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
    #     dataset = ALSCDataModule_(data_dir,  args.train['batch_size'], num_workers=0)   
    tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.setup_(tokenizer)
    dataset.tokenizer = tokenizer
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'token_type_ids': -1,
        'label': -1, 
    }

    model =BaseForALSC(
        args=args,
        dataset=dataset,
    ) 
    
    return model, dataset


class BaseForALSC(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.aspect_count(dataset.datas['train'])
        
        if args.model['arch'] == 'glove':
            from tokenizer import get_glove_model
            self.plm_model = get_glove_model(dataset.tokenizer, scale=args.model['scale'])
        else:
            if args.model['use_adapter']:
                from utils_adapter import auto_load_adapter
                self.plm_model = auto_load_adapter(args, plm=plm if plm is not None else args.model['plm'])
            else: self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        
        self.plm_pooler = PoolerAll(self.plm_model.config)    
        self.hidden_size = self.plm_model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.loss_ce = nn.CrossEntropyLoss() 

        # self.bank = [
        #     {'index': s['index'], 'polarity': s['polarity'][0:3], 'label': s['label'], 
        #      'retrieval': s['retrieval'], 'feature': None} 
        #     for s in dataset.datas['train']
        # ]
        self.bank = {
            'label': torch.tensor([s['label'] for s in dataset.datas['train']]),
            'aspect': [None for s in dataset.datas['train']],
            'sentence': [None for s in dataset.datas['train']],
        }
        self.use_cl = args.model['use_cl']
        self.weight = args.model['weight']
        self.ret_num = args.model['ret_num']

    def aspect_count(self, trainset):
        aspect_term = [sample['aspect'] for sample in trainset]
        aspect_dict = defaultdict(list)
        for i, asp in enumerate(aspect_term):
            aspect_dict[asp].append(i)
        self.aspect_index_dict = aspect_dict

    def encode(self, inputs, modes=['cls', 'asp', 'all']):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        plm_out = self.plm_model(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
            )
        hidden_states = self.plm_pooler(plm_out.last_hidden_state)
        hidden_states = self.dropout(hidden_states)
        #hidden_states = plm_out.last_hidden_state

        outputs = {}
        for mode in modes:
            if mode == 'cls': outputs['cls'] = hidden_states[:,0]
            if mode == 'all': outputs['all'] = hidden_states
            if mode == 'asp':
                token_type_ids = inputs['token_type_ids'] # token_type_ids 的平均值
                outputs['asp'] = torch.stack([torch.mean(hidden_states[bi][tmp.bool()], dim=0) for bi, tmp in enumerate(token_type_ids)])

        return outputs
    ## static retrieval
    def get_ret_features_(self, inputs, features=None, mode='store'):
        if mode == 'store': # 存储表示
            for idx, fea in zip(inputs['index'], features):         
                self.bank[idx]['feature'] = fea.detach().cpu()
        
        if mode == 'fetch': # 取出表示
            ret_pos, ret_neg = [], [] # [pos, negs]
            for idx in inputs['index']:
                ret_i, pos_i, negs_i = self.bank[idx.item()], [], []
                for lab, idxes in ret_i['retrieval'].items():
                    if lab == ret_i['polarity']:
                        pos_i.extend([self.bank[i]['feature'] for i in idxes[0:self.args.model['ret_num']]])
                    else: negs_i.extend([self.bank[i]['feature'] for i in idxes[0:self.args.model['ret_num']]])

                ret_pos.append(torch.stack(pos_i).mean(dim=0))
                ret_neg.append(torch.stack(negs_i).mean(dim=0))
                # ret_labels.append(torch.tensor([1]+[0]*len(negs_i)).type_as(inputs['label']))    
        
            return torch.stack(ret_pos), torch.stack(ret_neg)

    ## dynamic retrieval
    def get_ret_features(self, inputs, features=None, mode='store'):
        if mode == 'store': # 存储表示
            for i, idx in enumerate(inputs['index']):         
                self.bank['aspect'][idx] = features['asp'][i].detach().cpu()
                self.bank['sentence'][idx] = features['cls'][i].detach().cpu()
        
        if mode == 'fetch': # 取出表示
            ## 0. 找到相似 aspect
            aspect_train = torch.stack(self.bank['aspect']).type_as(features['asp'])
            sim_aspect = F.cosine_similarity(features['asp'].unsqueeze(dim=1), aspect_train.unsqueeze(dim=0), dim=-1)
            ret_pos, ret_neg = [], []
            for i, sim in enumerate(sim_aspect):
                pos_i, neg_i = [], []
                for lab in range(self.num_classes):
                    mark = (self.bank['label'] == lab).type_as(inputs['label']) # 排除掉 非目标 label
                    mark[inputs['index'][i]] = 0
                    sim_lab = sim * mark
                    sim_lab_order = torch.argsort(sim_lab, descending=True)
                    if inputs['label'][i] == lab: # 是正样本
                        pos_i.extend([self.bank['sentence'][k] for k in sim_lab_order[0:self.ret_num]])
                    else: neg_i.extend([self.bank['sentence'][k] for k in sim_lab_order[0:self.ret_num]])

                ret_pos.append(torch.stack(pos_i).mean(dim=0))
                ret_neg.append(torch.stack(neg_i).mean(dim=0))
                # ret_labels.append(torch.tensor([1]+[0]*len(negs_i)).type_as(inputs['label']))    
        
            return torch.stack(ret_pos).type_as(features['cls']), torch.stack(ret_neg).type_as(features['cls'])

    def cl_calculate(self, features, ret_features, labels, temp=1.0):
        cosine_sim = F.cosine_similarity(features.unsqueeze(dim=1), ret_features.unsqueeze(dim=0), dim=-1) / temp
        cosine_sim_exp = torch.exp(cosine_sim)
        cl_loss = torch.stack([-torch.log(cosine_sim_exp[l][l]/cosine_sim_exp[l].sum()) for l in labels])

        return cl_loss.mean()

    def forward(self, inputs, stage='train'):
        outputs = self.encode(inputs, modes=['cls', 'asp'])
        logits = self.classifier(outputs['cls']) 
        loss = self.loss_ce(logits, inputs['label'])

        ## 先存储 后计算
        if stage == 'train' and self.args.model['store_first']:
            self.get_ret_features(inputs, outputs, mode='store') # 存储表示

        if stage == 'train' and self.use_cl>0:
        ####################### 静态检索 #######################
        #     ret_pos, ret_neg = self.get_ret_features(inputs, mode='fetch') # 取出表示
        #     # com_features = torch.cat([features, ret_pos.type_as(features), ret_neg.type_as(features)], dim=0)
        #     com_features = torch.cat([ret_pos.type_as(features), ret_neg.type_as(features)], dim=0)
        #     cl_labels = torch.range(0, features.size(0)-1).type_as(inputs['label'])
        #     loss_cl = self.cl_calculate(features, com_features, cl_labels, temp=1.0)

        #     loss = loss*(1-self.weight) + loss_cl*self.weight
        ######################################################
            ret_pos, ret_neg = self.get_ret_features(inputs, outputs, mode='fetch') # 取出表示
            com_features = torch.cat([ret_pos, ret_neg], dim=0)
            cl_labels = torch.range(0, outputs['cls'].size(0)-1).type_as(inputs['label'])
            loss_cl = self.cl_calculate(outputs['cls'], com_features, cl_labels, temp=1.0)

            loss = loss*(1-self.weight) + loss_cl*self.weight
            # loss += loss_cl*self.weight
        
        ## 先计算 后存储
        if stage == 'train'and not self.args.model['store_first']:
            self.get_ret_features(inputs, outputs, mode='store') # 存储表示

        return {
            'idxs': inputs['index'].cpu(),
            'features':    outputs['cls'].detach().cpu(),
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': inputs['label'].cpu(),
        }
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.use_cl += 1

        output = self(batch, stage='train')
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }
