
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import DataCollatorWithPadding
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

class VietnameseSentimentAnalysis(Dataset):
    def __init__(self, dataset, tokenizer, rdrsegmenter = None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.rdrsegmenter == None:
          text = item['sentence']
        else :
          output = self.rdrsegmenter.word_segment(item['sentence'])
          text = ' '.join(output)
        label = item['sentiment']
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = self.swap_tensor_values(torch.tensor(label))
        return inputs

    def swap_tensor_values(self, input_tensor):
        """
        Swap values in the tensor:
        - Change every instance of 1 to 2
        - Change every instance of 2 to 1
        Args:
        - input_tensor (torch.Tensor): A tensor containing values 0, 1, and 2.

        Returns:
        - torch.Tensor: A tensor with swapped values.
        """
        # Create a copy of the input tensor to avoid modifying the original
        output_tensor = input_tensor.clone()

        # Perform the swapping
        output_tensor[output_tensor == 1] = 3  # Temporarily change 1 to 3
        output_tensor[output_tensor == 2] = 1  # Change 2 to 1
        output_tensor[output_tensor == 3] = 2  # Change 3 to 2

        return output_tensor

def apply_random_oversampling(dataset):
    df = pd.DataFrame(dataset)
    X = df['sentence']
    y = df['sentiment']

    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)

    df_resampled = pd.DataFrame({'sentence': X_resampled.flatten(), 'sentiment': y_resampled})
    return HFDataset.from_pandas(df_resampled)

def create_dataloaders(tokenizer, batch_size, rdrsegmenter = None):
    # Load and oversample the training dataset
    sentiment_dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    train_dataset = sentiment_dataset['train']
    train_dataset_resampled = apply_random_oversampling(train_dataset)

    # Load and oversample the validation dataset
    val_dataset = sentiment_dataset['validation']
    val_dataset_resampled = apply_random_oversampling(val_dataset)

    # Load test dataset
    test_dataset = sentiment_dataset['test']

    # Create Dataset objects
    train_dataset = VietnameseSentimentAnalysis(dataset=train_dataset_resampled, tokenizer=tokenizer, rdrsegmenter = rdrsegmenter)
    val_dataset = VietnameseSentimentAnalysis(dataset=val_dataset_resampled, tokenizer=tokenizer, rdrsegmenter = rdrsegmenter)
    test_dataset = VietnameseSentimentAnalysis(dataset=test_dataset, tokenizer=tokenizer, rdrsegmenter = rdrsegmenter)

    # Print lengths of datasets
    print(f'Train Length: {len(train_dataset)}')
    print(f'Validation Length: {len(val_dataset)}')
    print(f'Test Length: {len(test_dataset)}')

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.data_collator
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.data_collator
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.data_collator
    )

    return train_dataloader, val_dataloader, test_dataloader
