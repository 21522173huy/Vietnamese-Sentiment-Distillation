
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer

# Move to parent folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set manual seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_name', choices=['ViT5', 'Roberta-XLM'], type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()

    if args.teacher_name == 'ViT5':
      from model.t5_model import CustomModel

      print(f'USE T5 MODEL')
      teacher_model = CustomModel(t5_version = 'VietAI/vit5-large', num_labels = 3)
      tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-large')
      optimizer = torch.optim.AdamW(teacher_model.parameters(), weight_decay=0.01, lr = 2e-05)

    elif args.teacher_name == 'Roberta-XLM':
      from model.roberta_model import TeacherModel

      print(f'USE ROBERTA-XLM MODEL')
      teacher_model = TeacherModel(model_name = 'FacebookAI/xlm-roberta-large', num_labels = 3)
      tokenizer = teacher_model.tokenizer
      optimizer = torch.optim.Adam(params=teacher_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # Load the dataset
    from dataset import create_dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer = tokenizer,
                                                                           batch_size=args.batch_size,
                                                                            rdrsegmenter = None) # rdrsegmenter is used for PhoBert

    # Finetuning Config
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Finetuning
    from teacher.finetune_function import finetune_teacher
    train_loss, test_loss, train_metrics, test_metrics = finetune_teacher(teacher_model=teacher_model,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          scheduler=scheduler,
                                                                          save_path = f'teacher/checkpoints/{args.teacher_name}-Teacher-best.pt',
                                                                          max_grad_norm=1.0,
                                                                          epochs=args.epochs,
                                                                          patience=5)

    results = {
        'Train': {
            'Loss': train_loss,
            'Metrics': train_metrics,
        },
        'Val': {
            'Loss': test_loss,
            'Metrics': test_metrics,
        }
    }

    # # Save results to a JSON file
    # import json
    # id = args.teacher_name.split('/')[-1]
    # with open(f'{id}_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
