
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
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--student_type', choices=['large', 'base'], type=str, default = 'base')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--soft_weight', type=float)
    parser.add_argument('--hard_weight', type=float)
    args = parser.parse_args()

    if args.teacher_name == 'ViT5':
      from model.t5_model import CustomModel

      # Teacher
      teacher_model = CustomModel(t5_version = 'VietAI/vit5-large', num_labels = 3)
      tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-large')

      # Student
      if args.student_type == 'base':
        student_model = CustomModel(t5_version = 'VietAI/vit5-base', num_labels = 3)
      elif args.student_type == 'large':
        from model.t5_model import CustomT5_FromLarge
        student_model = CustomT5_FromLarge(num_labels = 3, num_blocks = 6)

      # Optimizer
      optimizer = torch.optim.AdamW(student_model.parameters(), weight_decay=0.01, lr = 2e-05)

    elif args.teacher_name == 'Roberta-XLM':
      from model.roberta_model import TeacherModel

      # Teacher
      teacher_model = TeacherModel(model_name = 'FacebookAI/xlm-roberta-large', num_labels = 3)
      tokenizer = teacher_model.tokenizer

      # Student
      student_model = TeacherModel(model_name = 'FacebookAI/xlm-roberta-base', num_labels = 3)
      # Optimizer
      optimizer = torch.optim.Adam(params=student_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # Load the dataset
    from dataset import create_dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer = tokenizer,
                                                                           batch_size=args.batch_size,
                                                                            rdrsegmenter = None) # rdrsegmenter is used for PhoBert

    # Finetuning Config
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Finetuning
    type_in_path = 'Base' if args.student_type == 'base' else 'Large'
    if args.student_type == 'Roberta-XLM' : type_in_path = 'Base'

    from student.training_function import training_student
    train_loss, test_loss, train_metrics, test_metrics = training_student(student_model=student_model,
                                                                          teacher_model=teacher_model,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          scheduler=scheduler,
                                                                          max_grad_norm=1.0,
                                                                          epochs=args.epochs,
                                                                          save_path = f'student/checkpoints/{type_in_path}-Student-{args.teacher_name}-{int(args.soft_weight * 100)}{int(args.hard_weight * 100)}-best.pt',
                                                                          patience=5,
                                                                          temperature=2,
                                                                          soft_weight=args.soft_weight,
                                                                          hard_weight=args.hard_weight)
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
