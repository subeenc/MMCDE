# Standard library
import argparse
import codecs
from datetime import datetime
import os
import pickle
import sys
from typing import List

# Third-party libraries
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Pool, set_start_method
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Local modules and custom files
from network import MMCDE
from metrics import *
from utils import load_config, setup_logging, pretrained_model_mapper, init_model, validate_saved_model
import data_provider


class Trainer:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        
        self.model = None
        self.best_dev_evaluation_result = EvaluationResult()  
        self.best_test_evaluation_result = EvaluationResult() 
        self.best_epoch = -1 
        
        self.disable_tqdm = False if self.args.local_rank in [-1, 0] else True
    
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        self.data_provider = data_provider.DataProvider(args)
        self.data_provider.init_data_socket()
    
    def load_model(self):
        self.args.num_labels = len(self.data_provider.get_labels())
        self.args.total_steps = self.data_provider.peek_num_train_examples()

        self.model = MMCDE(self.args)
        
        self.model = init_model(self.model, self.logger, self.args.init_checkpoint)
        if self.args.verbose:
            self.logger.info("==== Model initialized with weights from the PLATO checkpoint. ====")
                
        self.model.set_finetune()
        self.model = self.model.to(self.args.device)

    def eval_tasks(
        self,
        strategy: str = 'mean_by_role',
        role: str = 'all',
        tasks: List[str] = None,
        mode: str = 'test',
        force: bool = False,
        features: np.array = None
    ):

        if tasks is None:
            return

        if features is None:
            self.model.eval()
            test_loader = self.data_provider.get_clustering_test_loader(mode=mode)

            features = []
            with torch.no_grad():
                for step, batch in enumerate(test_loader):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    output_dict = self.model(data=batch, strategy=strategy)
                    feature = output_dict["final_feature"]
                    features.append(feature)
            features = torch.cat(features)
            cpu_features = features.cpu()
        else:
            cpu_features = torch.tensor(features)
            features = cpu_features.to(self.args.device)

        test_path = os.path.join(self.args.data_dir, "clustering_%s.tsv" % mode)
        with codecs.open(test_path, "r", "utf-8") as f:
            labels = [int(line.strip('\n').split("\t")[-1]) for line in f]
            
        with codecs.open(test_path, "r", "utf-8") as f:
            photo_indices = [line.strip('\n').split("\t")[1].split("|")[0] for line in f]
            
        best_evaluation_result = self.best_test_evaluation_result if mode == 'test' else self.best_dev_evaluation_result
        evaluation_result = EvaluationResult()
        if 'clustering' in tasks:
            n_average = max(3, 10 - features.shape[0] // 500)
            er = feature_based_evaluation_at_once(features=cpu_features,
                                                  labels=labels,
                                                  n_average=n_average,
                                                  tsne_visualization_output=None,
                                                  tasks=['clustering'],
                                                  dtype='float32',
                                                  logger=None,
                                                  note=','.join([mode, strategy, role]))
            evaluation_result.RI = er.RI
            evaluation_result.NMI = er.NMI
            evaluation_result.acc = er.acc
            evaluation_result.purity = er.purity
            
        if 'semantic_relatedness' in tasks or 'session_retrieval' in tasks:
            er = feature_based_evaluation_at_once(features=cpu_features,
                                                  labels=labels,
                                                  n_average=0,
                                                  tsne_visualization_output=None,
                                                  tasks=['semantic_relatedness', 'session_retrieval'],
                                                  dtype='float32',
                                                  logger=None,
                                                  note=','.join([mode, strategy, role]))

            evaluation_result.SR = er.SR
            evaluation_result.MRR = er.MRR
            evaluation_result.MAP = er.MAP
         
        # How is_best is updated: If at least two of purity, SR, and MAP have improved, assign is_best=True
        evaluation_scores = [evaluation_result.purity, evaluation_result.SR, evaluation_result.MAP]
        best_scores = [best_evaluation_result.purity, best_evaluation_result.SR, best_evaluation_result.MAP]
        better_cnt = sum(e > b for e, b in zip(evaluation_scores, best_scores))
        is_best = True if better_cnt >= 2 else False

        if self.args.verbose:
            print(f"\n=== mode: {mode}")
            print(evaluation_scores)
            print(best_scores)
            print(better_cnt, is_best)

        if 'align_uniform' in tasks:
            if is_best or mode == 'test' or force:
                er = feature_based_evaluation_at_once(features=cpu_features,
                                                    labels=labels,
                                                    gpu_features=features,
                                                    n_average=0,
                                                    tsne_visualization_output=None,
                                                    tasks=['align_uniform'],
                                                    dtype='float32',
                                                    logger=None,
                                                    note=','.join([mode, strategy, role]))

                evaluation_result.alignment = er.alignment
                evaluation_result.adjusted_alignment = er.adjusted_alignment
                evaluation_result.uniformity = er.uniformity

        if self.args.verbose:
            evaluation_result.show(logger=self.logger,
                                note=','.join([mode, strategy, role]))

        return is_best, evaluation_result
            
    def train(self):
        if self.args.verbose:
            self.logger.info(f"Starting batch index {self.args.starting_batch_idx}")
            self.logger.info("device: %s n_gpu: %s" % (self.args.device, self.args.n_gpu))
            
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)

        # DDP
        self.model = DistributedDataParallel(self.model, 
                                        device_ids=[self.args.local_rank],
                                        output_device=self.args.local_rank,)
        
        self.model.train()
        global_step = 0
        start = datetime.now()

        train_loader = self.data_provider.get_train_loader()
        for epoch in range(int(self.args.num_train_epochs)):
            with tqdm(total=train_loader.__len__() * self.args.train_batch_size, ncols=90, disable=self.disable_tqdm) as pbar:
                for step, batch in enumerate(train_loader):
                    if self.args.local_rank != -1:
                        train_loader.sampler.set_epoch(epoch)

                    batch = tuple(t.to(self.args.device) for t in batch)
                    output_dict = self.model(data=batch, strategy='mean_by_role')

                    loss = output_dict['loss']
                    if self.args.n_gpu > 1:
                        loss = loss.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix(epoch=epoch, global_step=global_step, train_loss=float(loss.item()))
                    
                    # Periodically evaluate the model on the dev dataset
                    global_step += 1
                    if global_step % self.args.eval_interval == 0:
                        if self.args.verbose:
                            self.logger.info(f'***** Epoch = {epoch}, Global Step = {global_step}: Evaluating on dev *****')
                        
                        # Evaluate on dev dataset
                        is_best_dev, dev_evaluation_result = self.eval_tasks(strategy='mean_by_role',
                                                                         tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                                                                         mode='dev',
                                                                         force=True)

                        # If dev performance improves, evaluate on the test dataset
                        if is_best_dev:
                            if self.args.verbose:
                                self.logger.info(f"New best dev result at epoch {epoch}, global step {global_step}. Evaluating on test set.")
                                
                            self.best_epoch = epoch
                            
                            # Evaluate on test dataset
                            is_best_test, test_evaluation_result = self.eval_tasks(strategy='mean_by_role',
                                                                                    tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                                                                                    mode='test')
                            
                            # Save the model if test performance improves
                            if is_best_test:
                                self.best_epoch = epoch
                                self.best_model_name = os.path.join(self.args.output_dir, self.args.best_model)
                                if self.args.local_rank in [-1, 0]:
                                    torch.save(self.model.module.state_dict(), self.best_model_name)
                                    self.logger.info(f"Best model saved to {self.best_model_name}")
                                    
                                    # Update the best dev and test evaluation results
                                    self.best_dev_evaluation_result = dev_evaluation_result
                                    self.best_test_evaluation_result = test_evaluation_result
                        
                        # Switch back to training mode
                        self.model.train()
                    
                    pbar.update(self.args.train_batch_size)
                    
            if self.args.verbose:
                self.logger.info(f"Epoch [{epoch+1}/{self.args.num_train_epochs}], Loss: {loss.item()}")

        if self.args.verbose:
            self.logger.info(f"Training completed in: {datetime.now() - start}")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
      
def main():
    dist.init_process_group(backend='nccl')  # 'nccl', 'gloo', 'mpi'
    verbose = dist.get_rank() == 0  # Configure logging to be displayed only on rank 0
    
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Training configuration")
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'], help="random seed")
    parser.add_argument("--init_checkpoint", type=str, default=config['train']['init_checkpoint'], help='checkpoint path of initializing backbone.')
    parser.add_argument('--num_train_epochs', type=int, default=config['train']['num_train_epochs'], help="Number of epochs to train")
    parser.add_argument('--train_batch_size', type=int, default=config['train']['train_batch_size'], help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'], help="Learning rate for training")
    # test & dev
    parser.add_argument("--stage", type=str, default=config['eval']['stage'], help='[train, test, dev]')
    parser.add_argument('--test_batch_size', type=int, default=config['eval']['test_batch_size'], help="Batch size for testing")
    parser.add_argument('--dev_batch_size', type=int, default=config['eval']['dev_batch_size'], help="Batch size for dev")
    parser.add_argument('--eval_interval', type=int, default=config['eval']['eval_interval'], help="Interval for evaluation during training")
    # log
    parser.add_argument('--log_dir', type=str, default=config['logging']['log_dir'], help="Directory for logging")
    parser.add_argument('--log_file', type=str, default=config['logging']['log_file'], help="Directory for logging")
    parser.add_argument('--log_level', type=str, default=config['logging']['log_level'], help="Logging level")
    # model
    parser.add_argument("--backbone", type=str, default=config['model']['backbone'], help='Options: []')
    parser.add_argument("--config_file", default="model/plato/config.json", type=str)
    parser.add_argument("--temperature", default=config['model']['temperature'], type=float)
    parser.add_argument("--local_loss_rate", default=config['model']['local_loss_rate'], type=float)
    parser.add_argument('--output_dir', type=str, default=config['model']['output_dir'], help="Directory for output")
    parser.add_argument('--best_model', type=str, default=config['model']['output_dir'], help="Best Model")
    # data
    parser.add_argument('--data_dir', type=str, default=config['data']['data_dir'], help="Directory for dataset")
    parser.add_argument('--dataset_name', type=str, default=config['data']['dataset_name'], help="Name of dataset")
    parser.add_argument('--img_dir', type=str, default=config['data']['img_dir'], help="Directory for images")
    parser.add_argument('--use_image_tensors', type=str2bool, default=config['data']['use_image_tensors'], help="Set to 'True' to use pre-saved image tensors from a dictionary, or 'False' to process raw images directly.")
    parser.add_argument('--image_tensor_dir', type=str, default=config['data']['image_tensor_dir'], help="Directory for image features")
    parser.add_argument('--max_lines', type=int, default=config['data']['max_lines'], help="Number of lines to read from the training data")
    parser.add_argument('--starting_batch_idx', type=int, default=config['data']['starting_batch_idx'], help="Number of samples per file to load")
    parser.add_argument('--batch_per_file', type=int, default=config['data']['batch_per_file'], help="Number of samples per file to load")
    parser.add_argument('--line_sep_token', type=str, default=config['data']['line_sep_token'], help="Directory for dataset")
    parser.add_argument('--sample_sep_token', type=str, default=config['data']['sample_sep_token'], help="Directory for dataset")
    parser.add_argument('--turn_sep_token', type=str, default=config['data']['turn_sep_token'], help="Directory for dataset")
    parser.add_argument("--use_sep_token", type=str2bool, default=config['data']['use_sep_token'])
    parser.add_argument('--num_neg_samples', type=int, default=config['data']['num_neg_samples'], help="Number of negative samples")
    parser.add_argument('--num_all_samples', type=int, default=config['data']['num_all_samples'], help="Number of all samples")
    parser.add_argument('--max_context_length', type=int, default=config['data']['max_context_length'], help="Max context length")
    parser.add_argument('--max_seq_length', type=int, default=config['data']['max_seq_length'], help="Max sequence length")
    #DDP
    parser.add_argument("--backend", type=str, default=config['ddp']['backend'], help="local rank for DDP training.")
    parser.add_argument("--world_size", type=int, default=config['ddp']['world_size'], help="local rank for DDP training.")
    parser.add_argument("--local_rank", type=int, default=os.environ.get('LOCAL_RANK', 0), help="local rank for DDP training.")

    args = parser.parse_args()
       
    logger = setup_logging(args.log_dir, args.log_file, args.log_level)
    args.logger = logger

    # Get pretrained model name based on backbone
    pretrained_model_name = pretrained_model_mapper(args.backbone)
    args.pretrained_model_name = pretrained_model_name
    
    # device info
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch.cuda.set_device(args.local_rank)
    
    args.verbose = verbose
    if args.verbose:
        args.logger.info(f"Training configuration: {args}")
        args.logger.info("device: %s n_gpu: %s" % (args.device, args.n_gpu))
    
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # train
    trainer = Trainer(args)
    trainer.load_model()

    if args.stage == "train":
        pre_train_time = time()
        trainer.train()
        args.logger.info('Total time costs: %s mins' % ((time() - pre_train_time) / 60))
    elif args.stage == "test":
        pre_test_time = time()
        args.logger.info(f'Running test with backbone: {pretrained_model_name}')
        trainer.eval_tasks(strategy='mean_by_role',
                           tasks=['clustering', 'semantic_relatedness', 'session_retrieval', 'align_uniform'],
                           mode='test')
        args.logger.info('Total time costs: %s mins' % ((time() - pre_test_time) / 60))


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    
    main()


