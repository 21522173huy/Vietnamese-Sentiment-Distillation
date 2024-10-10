
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Callable, Any, Tuple
from torch import nn

class TeacherModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(TeacherModel, self).__init__()
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trainable = True
        self.load_in_nbits = 16  # Adjust as needed

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.teacher_model.to(self.device)
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    def save_checkpoint(self, save_path: str):
        self.teacher_model.save_pretrained(save_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
