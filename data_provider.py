#!/usr/bin/python
# _*_coding:utf-8_*_
import os
import codecs
import ast
from multiprocessing import Pool
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from transformers import AutoTokenizer
from model.plato.configuration_plato import PlatoConfig


class BertExample():
    def __init__(self, guid, role, photoidx, photoid, text_a, img_feature, text_b=None, label=None): 
        self.guid = guid
        self.role = role
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.photoidx = photoidx 
        self.photoid = photoid 
        self.img_feature = img_feature  


class BertFeatures():
    def __init__(self, input_ids, input_mask, segment_ids, role_ids, label_id, photo_ids, img_feature, turn_ids=None, position_ids=None, guid=None): 
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.role_ids = role_ids
        self.turn_ids = turn_ids
        self.position_ids = position_ids
        self.label_id = label_id
        self.guid = guid
        self.photo_ids = photo_ids 
        self.img_feature = img_feature 
        self.batch_size = len(self.input_ids)
        

class DataProvider():
    def __init__(self, args):
        self.tokenizer = None
        self.train_loader = None
        self.num_train_examples = None
        self.clustering_test_loader = None
        self.clustering_dev_loader = None
        self.tss_test_loader = None
        self.num_workers = 20
        self.args = args
        self.logger = args.logger

    def init_data_socket(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name)
        self.tokenizer_config = PlatoConfig.from_json_file(self.args.config_file)
        self.tokenizer_config.max_seq_length = self.tokenizer_config.max_position_embeddings
        self.labels_list = ["0", "1"]

    def get_tokenizer(self):
        return self.tokenizer

    def get_labels(self):
        return self.labels_list

    def peek_num_train_examples(self):
        self.num_train_examples = self.line_statistics(self.args.data_dir + "/train.tsv")
        return self.num_train_examples
    
    def line_statistics(self, file_name):
        if file_name is None:
            return 0

        content = os.popen("wc -l %s" % file_name)
        line_number = int(content.read().split(" ")[0])
        return line_number

    def load_data(self, data_file, mode):
        all_img_features = torch.load(f"{self.args.data_dir}/{mode}_img_features.pt")
        if self.args.verbose:
            self.args.logger.info(f"Load {mode} image features")
        
        with codecs.open(data_file, "r", "utf-8") as f_in:
            bert_examples = []
            for i, line in enumerate(f_in):
                line_array = [s.strip() for s in line.split(self.args.line_sep_token) if s.strip()]

                role, session, label, photoidx, photoid = line_array[0], line_array[2], line_array[4], line_array[1], line_array[3]
                photoid = ast.literal_eval(photoid)
                bert_examples.append(BertExample(guid=None, role=role, text_a=session, label=label, photoidx=photoidx, photoid=photoid, img_feature=all_img_features[i]))
                
        return bert_examples

    def convert_examples_worker(self, worker_index, start_index, end_index, examples):
        return self.__convert_examples_worker_for_plato(worker_index, start_index, end_index, examples)

    def __convert_examples_worker_for_plato(self, worker_index, start_index, end_index, examples):
        features = []
        self.logger.debug("converting_examples, worker_index: %s start: %s end: %s" % (worker_index, start_index, end_index))

        for data_index, example in enumerate(examples):
            if data_index < start_index or data_index >= end_index:
                continue
            
            sample_list = example.text_a.split(self.args.sample_sep_token)
            role_list = example.role.split(self.args.sample_sep_token)
            role_list = [[int(r) for r in role] for role in role_list]
            
            photoidx_list = example.photoidx.split(self.args.sample_sep_token)
            photoidx_list = [[int(p) for p in photoidx] for photoidx in photoidx_list]

            sample_input_ids = []
            sample_segment_ids = []
            sample_role_ids = []
            sample_input_mask = []
            sample_turn_ids = []
            sample_position_ids = []
            sample_photo_ids = []  
            sample_image_features = torch.squeeze(torch.stack(example.img_feature), dim=1)

            for t, s in enumerate(sample_list):
                text_tokens = []
                text_turn_ids = []
                text_role_ids = []
                text_segment_ids = []
                text_photo_ids = []

                text_list = s.split(self.args.turn_sep_token)

                # token: token [eou] token [eou] [bos] token [eos]
                # role:   0     0     1     1     0     0      0
                # turn:   2     2     1     1     0     0      0
                # pos:    0     1     0     1     0     1      2
                
                # bou: begin of utterance
                # eou: end of utterance
                # bos: begin of sentence
                # eos: end of sentence
                bou, eou, bos, eos = "[unused0]", "[unused1]", "[unused0]", "[unused1]"

                # use [CLS] as the latent variable of PLATO
                # text_list[0] = self.args.start_token + ' ' + text_list[0]

                context = text_list
                response_tokens, response_role_ids, response_turn_ids, response_segment_ids, response_photo_ids = [], [], [], [], []

                # limit the context length
                context = context[-self.args.max_context_length:]
                
                current_role_list = role_list[t]
                current_photoidx_list = photoidx_list[t]

                for i, text in enumerate(context): 
                    word_list = self.tokenizer.tokenize(text)
                    uttr_len = len(word_list)

                    end_token = eou

                    role_id, turn_id = current_role_list[i], len(context) - i
                    photo_idx = current_photoidx_list[i]

                    text_tokens.extend(word_list + [end_token])
                    text_role_ids.extend([role_id] * (uttr_len + 1))
                    text_turn_ids.extend([turn_id] * (uttr_len + 1))
                    text_segment_ids.extend([0] * (uttr_len + 1))
                    text_photo_ids.extend([photo_idx] * (uttr_len + 1))

                text_tokens.extend(response_tokens)
                text_role_ids.extend(response_role_ids)
                text_turn_ids.extend(response_turn_ids)
                text_segment_ids.extend(response_segment_ids)
                text_photo_ids.extend(response_photo_ids)
                
                if len(text_tokens) > self.tokenizer_config.max_seq_length:
                    text_tokens = text_tokens[:self.tokenizer_config.max_seq_length]
                    text_turn_ids = text_turn_ids[:self.tokenizer_config.max_seq_length]
                    text_role_ids = text_role_ids[:self.tokenizer_config.max_seq_length]
                    text_segment_ids = text_segment_ids[:self.tokenizer_config.max_seq_length]
                    text_photo_ids = text_photo_ids[:self.tokenizer_config.max_seq_length]

                assert (max(text_turn_ids) <= self.args.max_context_length)

                text_position_ids = []
                text_position_id = 0
                for i, turn_id in enumerate(text_turn_ids):
                    if i != 0 and turn_id < text_turn_ids[i - 1]: 
                        text_position_id = 0
                    text_position_ids.append(text_position_id)
                    text_position_id += 1

                text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
                text_input_mask = [1] * len(text_input_ids)

                # Zero-pad up to the sequence length.
                while len(text_input_ids) < self.tokenizer_config.max_seq_length:
                    text_input_ids.append(0)
                    text_turn_ids.append(0)
                    text_role_ids.append(0)
                    text_segment_ids.append(0)
                    text_position_ids.append(0)
                    text_input_mask.append(0)
                    text_photo_ids.append(0)

                assert len(text_input_ids) == self.tokenizer_config.max_seq_length
                assert len(text_turn_ids) == self.tokenizer_config.max_seq_length
                assert len(text_role_ids) == self.tokenizer_config.max_seq_length
                assert len(text_segment_ids) == self.tokenizer_config.max_seq_length
                assert len(text_position_ids) == self.tokenizer_config.max_seq_length
                assert len(text_input_mask) == self.tokenizer_config.max_seq_length
                assert len(text_photo_ids) == self.tokenizer_config.max_seq_length

                sample_input_ids.append(text_input_ids)
                sample_turn_ids.append(text_turn_ids)
                sample_role_ids.append(text_role_ids)
                sample_segment_ids.append(text_segment_ids)
                sample_position_ids.append(text_position_ids)
                sample_input_mask.append(text_input_mask)
                sample_photo_ids.append(text_photo_ids)
                
            label_id = [1, 1] + [0] * self.args.num_neg_samples
            bert_feature = BertFeatures(input_ids=sample_input_ids,
                                        input_mask=sample_input_mask,
                                        segment_ids=sample_segment_ids,
                                        role_ids=sample_role_ids,
                                        turn_ids=sample_turn_ids,
                                        position_ids=sample_position_ids,
                                        label_id=label_id,
                                        photo_ids=sample_photo_ids,
                                        img_feature=sample_image_features)

            features.append(bert_feature)
        
        return features

    def convert_examples_to_features(self, examples):
        worker_results = []
        features = []
        num_workers = self.num_workers

        pool = mp.Pool(processes=num_workers)
        partition_size = math.ceil(len(examples) / num_workers)

        for i in range(num_workers):
            start = i * partition_size
            end = min((i + 1) * partition_size, len(examples))
            worker_results.append(pool.apply_async(self.convert_examples_worker, args=(i, start, end, examples)))
            if end == len(examples):
                break

        for processor in worker_results:
            feature_list = processor.get()
            features.extend(feature_list)
                
        pool.close()
        pool.join()

        return features

    def get_train_loader(self):
        if self.train_loader is not None:
            return self.train_loader

        bert_examples = self.load_data(self.args.data_dir + "/train.tsv", mode="train")
        if self.args.verbose:
            self.args.logger.info(f"Loaded {len(bert_examples)} training examples from train.tsv")

        bert_features = self.convert_examples_to_features(bert_examples)
        if self.args.verbose:
            self.args.logger.info(f"Converted {len(bert_examples)} examples into {len(bert_features)} features")

        self.num_train_steps = int(len(bert_examples) / self.args.train_batch_size * self.args.num_train_epochs)

        if self.args.verbose:
            self.logger.info("***** Running training *****")
            self.logger.info("  Num examples = %d", self.num_train_examples)
            self.logger.info("  Batch size = %d", self.args.train_batch_size)
            self.logger.info("  Num steps = %d", self.num_train_steps)
        
        all_input_ids = torch.stack([torch.tensor(f.input_ids) for f in bert_features], dim=0) 
        all_input_mask = torch.stack([torch.tensor(f.input_mask) for f in bert_features], dim=0)
        all_segment_ids = torch.stack([torch.tensor(f.segment_ids) for f in bert_features], dim=0)
        all_role_ids = torch.stack([torch.tensor(f.role_ids) for f in bert_features], dim=0)
        all_turn_ids = torch.stack([torch.tensor(f.turn_ids) for f in bert_features], dim=0)
        all_position_ids = torch.stack([torch.tensor(f.position_ids) for f in bert_features], dim=0)
        all_label_ids = torch.stack([torch.tensor(f.label_id) for f in bert_features], dim=0)
        all_photo_ids = torch.stack([torch.tensor(f.photo_ids) for f in bert_features], dim=0)
        all_image_features = torch.stack([f.img_feature for f in bert_features], dim=0)

        train_data = TensorDataset(all_input_ids,
                                   all_input_mask,
                                   all_segment_ids,
                                   all_role_ids,
                                   all_turn_ids,
                                   all_position_ids,
                                   all_label_ids,
                                   all_photo_ids,
                                   all_image_features)

        train_sampler = RandomSampler(train_data) if self.args.local_rank == -1 else DistributedSampler(train_data,
                                                                                                        num_replicas=torch.cuda.device_count(),
                                                                                                       rank=self.args.local_rank)
        if self.args.verbose:
            self.logger.info("***** DataLoader *****")
        self.train_loader = DataLoader(train_data,
                                       sampler=train_sampler,
                                       batch_size=self.args.train_batch_size,
                                       num_workers=2)
        
        return self.train_loader

    def get_clustering_test_loader(self, mode='test', level='dialogue'):
        if level == 'dialogue':
            if mode == 'test' and self.clustering_test_loader is not None:
                return self.clustering_test_loader
            if mode == 'dev' and self.clustering_dev_loader is not None:
                return self.clustering_dev_loader

            bert_examples = self.load_data(self.args.data_dir + "/clustering_%s.tsv" % mode, mode)
            if self.args.verbose:
                self.args.logger.info(f"Loaded {len(bert_examples)} {mode} examples from {mode}.tsv")    
        else:
            bert_examples = self.load_data_for_simcse(self.args.data_dir + "/clustering_%s.tsv" % mode)
            
        bert_features = self.convert_examples_to_features(bert_examples)
        if self.args.verbose:
            self.args.logger.info(f"Converted {len(bert_examples)} examples into {len(bert_features)} features")

        all_input_ids = torch.stack([torch.tensor(f.input_ids) for f in bert_features], dim=0)
        all_input_mask = torch.stack([torch.tensor(f.input_mask) for f in bert_features], dim=0)  
        all_segment_ids = torch.stack([torch.tensor(f.segment_ids) for f in bert_features], dim=0)
        all_role_ids = torch.stack([torch.tensor(f.role_ids) for f in bert_features], dim=0)
        all_turn_ids = torch.stack([torch.tensor(f.turn_ids) for f in bert_features], dim=0)
        all_position_ids = torch.stack([torch.tensor(f.position_ids) for f in bert_features], dim=0)
        all_label_ids = torch.stack([torch.tensor(f.label_id) for f in bert_features], dim=0)
        all_photo_ids = torch.stack([torch.tensor(f.photo_ids) for f in bert_features], dim=0)
        all_image_features = torch.stack([f.img_feature for f in bert_features], dim=0)
        
        if level == 'dialogue':
            test_data = TensorDataset(all_input_ids,
                                      all_input_mask,
                                      all_segment_ids,
                                      all_role_ids,
                                      all_turn_ids,
                                      all_position_ids,
                                      all_label_ids,
                                      all_photo_ids,
                                      all_image_features)
        else:
            all_guids = torch.tensor([f.guid for f in bert_features], dtype=torch.int)
            test_data = TensorDataset(all_input_ids,
                                      all_input_mask,
                                      all_segment_ids,
                                      all_role_ids,
                                      all_turn_ids,
                                      all_position_ids,
                                      all_label_ids,
                                      all_photo_ids,
                                      all_image_features,
                                      all_guids)
        
        test_sampler = SequentialSampler(test_data)
        if mode == 'test':
            self.clustering_test_loader = DataLoader(test_data,
                                                     sampler=test_sampler,
                                                     batch_size=self.args.test_batch_size)
            return self.clustering_test_loader
        elif mode == 'dev':
            self.clustering_dev_loader = DataLoader(test_data,
                                                     sampler=test_sampler,
                                                     batch_size=self.args.dev_batch_size)
            return self.clustering_dev_loader
        else:
            raise ValueError('Unknown dataset mode: [%s]' % mode)

