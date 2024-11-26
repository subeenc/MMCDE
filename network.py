import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.plato.configuration_plato import PlatoConfig
from model.plato.modeling_plato import PlatoModel
from transformers import AutoModel, AutoConfig, CLIPConfig, CLIPTextModel, CLIPProcessor

from info_nce import InfoNCE, info_nce

class BertAVG(nn.Module):
    def __init__(self, eps=1e-12):
        super(BertAVG, self).__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (torch.sum(m, dim=1, keepdims=True) + self.eps)

        avg_output = reduce_mean(hidden_states, attention_mask)
        return avg_output

    def equal_forward(self, hidden_states, attention_mask):
        mul_mask = hidden_states * attention_mask.unsqueeze(-1)
        avg_output = torch.sum(mul_mask, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + self.eps)
        return avg_output


class MMCDE(nn.Module):
    def __init__(self, args):
        super(MMCDE, self).__init__()
        self.args = args
        self.result = {}
        num_labels, total_steps = args.num_labels, args.total_steps

        self.config = PlatoConfig.from_json_file(self.args.config_file)
        self.bert = PlatoModel(self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = BertAVG(eps=1e-6)
        self.logger = args.logger
        
        # Define image layers
        self.mapping_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU()
        )
        
        args.feature_dim = 768

        # Weight initialization for all layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def set_finetune(self):
        self.logger.debug("******************")
        name_list = ["11", "10", "9", "8", "7", "6"]
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for s in name_list:
                if s in name:
                    self.logger.debug(name)
                    param.requires_grad = True
    
    def forward(self, data, strategy='mean_by_role', output_attention=False):
        input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, photo_ids, image_features = data
        
        # [batch_size*10, 512]
        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])  
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])
        photo_ids = photo_ids.view(photo_ids.size()[0] * photo_ids.size()[1], photo_ids.size()[-1])
        image_features = image_features.view(image_features.size()[0] * image_features.size()[1], image_features.size()[-1])
        
        text_embeddings, text_pooled_output = self.text_encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)  # torch.Size([50, 512, 768]), torch.Size([50, 768])
        image_embeddings = self.mapping_layer(image_features)   # for alignment with text features
        
        # -------------------------- Global loss: Positive and negative pairs based on dialogue with order considered -------------------------- 
        # Step 1: Process text embeddings
        global_text_embeddings = text_embeddings * attention_mask.unsqueeze(-1)  # Apply attention mask to text embeddings
        global_text_embeddings = self.avg(global_text_embeddings, attention_mask)  # Aggregate text embeddings by dialogue (768 dimensions)

        # Step 2: Reshape embeddings for batch processing
        global_text_embeddings = global_text_embeddings.view(-1, self.args.num_all_samples, self.config.hidden_size)  # [batch_size, num_samples, hidden_size]
        global_image_embeddings = image_embeddings.view(-1, self.args.num_all_samples, self.config.hidden_size)  # [batch_size, num_samples, hidden_size]

        # Step 3: Concatenate text and image embeddings
        concat_all_embeddings = torch.cat((global_text_embeddings, global_image_embeddings), dim=-1)  # [batch_size, num_samples, hidden_size*2]

        # Step 4: Compute global loss
        logits_global = [] 
        labels_global = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0], device=self.args.device)  
        for i in range(1, self.args.num_all_samples): 
            cos_global = self.calc_cos(concat_all_embeddings[:, 0, :], concat_all_embeddings[:, i, :])  
            logits_global.append(cos_global)  
        logits_global = torch.stack(logits_global, dim=1)  
        our_loss_global = self.calc_loss(logits_global, labels_global)  

        # -------------------------- Local loss: image-text pair --------------------------
        # Step 1: Normalize embeddings for local loss
        l2_text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)  # Normalize text embeddings with L2 norm
        l2_image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # Normalize image embeddings with L2 norm

        # Step 2: Apply attention mask to normalized embeddings
        l2_text_embeddings = l2_text_embeddings * attention_mask.unsqueeze(-1)
        l2_text_embeddings = self.avg(l2_text_embeddings, attention_mask)

        # Step 3: Reshape normalized embeddings for batch processing
        l2_text_embeddings = l2_text_embeddings.view(-1, self.args.num_all_samples, self.config.hidden_size)  
        l2_image_embeddings = l2_image_embeddings.view(-1, self.args.num_all_samples, self.config.hidden_size)  

        # Step 4: Compute local loss
        logits_local = []  
        for i in range(0, self.args.num_all_samples): 
            cos_local = self.calc_cos(l2_text_embeddings[:, i, :], l2_image_embeddings[:, i, :]) 
            logits_local.append(cos_local) 
        logits_local = torch.stack(logits_local, dim=1)  
        our_loss_local = self.calc_loss(logits_local, labels) 

        # Combine global and local losses
        total_loss = (1-self.args.local_loss_rate)*our_loss_global + self.args.local_loss_rate*our_loss_local
        
        output_dict = {'loss': total_loss,
                       'final_feature': final_embedding}

        return output_dict

    def text_encoder(self, *x):
        input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids = x
        output = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            turn_ids=turn_ids,
                            role_ids=role_ids,
                            return_dict=True)
        
        all_output = output['hidden_states']
        pooler_output = output['pooler_output']
            
        return all_output[-1], pooler_output

    def calc_cos(self, x, y):
        cos_normalized  = torch.cosine_similarity(x, y, dim=1)
        cos = cos_normalized / self.args.temperature
        return cos
    
    def calc_loss(self, pred, labels):
        loss = -torch.mean(self.log_softmax(pred) * labels)
        return loss