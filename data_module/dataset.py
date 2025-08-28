import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import logging 
from pathlib import Path

# internal imports
from data_module.audio_processor import AudioProcessor

# logging setup
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing for `FLEURS` dataset (audio-text) pairs.

    Expected CSV format: 
    audio_path, transcription, duration
    """
    def __init__(
            self, 
            csv_path: str, 
            tokenizer, 
            audio_processor: Optional[AudioProcessor] = None,
            max_text_length: int = 512,
            audio_column: str = "audio_path",
            text_column: str = "transcription",
            cache_audio: bool = False, # whether to cache processed audio
    ):
        """
        Initializes the AudioDataset.

        Args:
            csv_path (str): Path to the CSV file containing dataset information.
            tokenizer: Tokenizer for processing text data.
            audio_processor (AudioProcessor, optional): Audio processor for handling audio data. If None, a default will be created.
            max_text_length (int): Maximum length for tokenized text sequences.
            audio_column (str): Name of the column in the CSV that contains audio file paths.
            text_column (str): Name of the column in the
            cache_audio (bool): Whether to cache processed audio data in memory.
        """

        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.audio_column = audio_column
        self.text_column = text_column
        self.cache_audio = cache_audio

        # Intialize Audio Processor
        if audio_processor is None:
            self.audio_processor = AudioProcessor()
        else:
            self.audio_processor = audio_processor

        # Load CSV data
        self.df = self._load_csv()
        self.audio_cache = {} if cache_audio else None

        logger.info(f"Loaded {len(self.df)} samples from {self.csv_path}")

    def _load_csv(self) -> pd.DataFrame:
        """
        Loads and validates the CSV data.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)

        # Validate required columns
        required_columns = [self.audio_column, self.text_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # clean text data (strip whitespace)
        df[self.text_column] = df[self.text_column].astype(str).str.strip()

        # Filtering out empty transcriptions
        initial_count = len(df)
        df = df[df[self.text_column].str.len() > 0].reset_index(drop=True)

        if len(df) < initial_count:
            logger.warning(f"Filtered out {initial_count - len(df)} samples with empty transcriptions.")

        return df
    
    def __len__(self) -> int: 
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing processed audio and tokenized text.
        """
        if idx >=len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
        
        row = self.df.iloc[idx]
        audio_path = row[self.audio_column] 
        texts = row[self.text_column]

        # Process audio
        audio = self._get_audio(audio_path, idx)
        if audio is None:
            # Return zero tensors if audio processing fails
            audio = torch.zeros(self.audio_processor.max_samples)
            logger.warning(f"Returning zero tensor for audio at index {idx} due to processing failure.")

        # Tokenize text
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" 
        )

        return {
            "audio": audio,
            "input_ids": tokenized["input_ids"].squeeze(0),  # remove batch dimension
            "attention_mask": tokenized["attention_mask"].squeeze(0), 
            "labels": tokenized["input_ids"].squeeze(0),  # for language modeling
            "texts": texts,
            "audio_path": str(audio_path)
        }
    

    def _get_audio(self, audio_path: str, idx: int) -> Optional[torch.Tensor]:
        """
        Get Processed audio, with optional caching. 
        """
        if self.cache_audio and idx in self.audio_cache: 
            return self.audio_cache[idx]
        
        audio = self.audio_processor.preprocess_single(audio_path)

        if self.cache_audio and audio is not None: 
            self.audio_cache[idx] = audio

        return audio
    
def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function to combine a list of samples into a batch.
    Stacks Audio and tokenized text tensors.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing batched audio and tokenized text tensors.
    """
    audio = torch.stack([item["audio"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])    
    labels = torch.stack([item["labels"] for item in batch])    
    texts = [item["transcription"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    return {
        "audio": audio, 
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": texts,
        "audio_paths": audio_paths
    }