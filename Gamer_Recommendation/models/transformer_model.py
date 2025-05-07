import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerScorer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.pooler_output
        score = self.scorer(cls_embedding)
        return score.squeeze()