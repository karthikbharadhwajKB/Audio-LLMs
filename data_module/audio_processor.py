import librosa
import numpy as np
import torch 
from typing import Optional, Union, List
import logging 

# logging setup
logger = logging.getLogger(__name__)

class AudioProcessor: 
    ""
    def __init__(self, 
                 sample_rate: int = 16_000, 
                 max_duration: float = 30.0, 
                 min_duration: float = 0.1, 
                 normalize: bool = True,
                 pad_mode: str = 'constant'
    ): 
        """
        Initializes the audio processor with specified parameters.
        Args:
            sample_rate (int): Target sample rate for audio files.
            max_duration (float): Maximum duration (in seconds) for audio files.
            min_duration (float): Minimum duration (in seconds) for audio files.
            normalize (bool): Whether to normalize audio samples to [-1, 1].
            pad_mode (str): Padding mode to use when padding audio samples.
        """
        
        self.sample_rate  = sample_rate 
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize    = normalize
        self.pad_mode     = pad_mode

        # pre-compute sample counts
        self.max_samples = int(self.sample_rate * self.max_duration)
        self.min_samples = int(self.sample_rate * self.min_duration)

        logger.info(f"AudioProcessor initialized: sample_rate={self.sample_rate},"
                    f"max_duration={self.max_duration}s, ({self.max_samples} samples)")

    def load_audio(self, audio_path: Union[str, bytes]) -> Optional[np.ndarray]:
        """
        Load and resample audio file.

        Args:
            audio_path (str or bytes): Path to the audio file.

        Returns:
            np.ndarray: Loaded audio samples, or None if loading fails.
        """
        try: 
            # Load with librosa (handles many formats)
            audio, sr = librosa.load(
                    audio_path, 
                    sr=self.sample_rate,
                    mono=True,
                    dtype=np.float32
            )   

            if len(audio) < self.min_samples: 
                logger.warning(f"Audio file {audio_path} is too short ({len(audio)/self.sample_rate:.2f}s).")
                return None
            
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return None 
        

    def preprocess_single(self, audio_path: Union[str, bytes]) -> Optional[torch.Tensor]:
        """
        Load and preprocess a single audio file.

        Args:
            audio_path (str or bytes): Path to the audio file.

        Returns:
            torch.Tensor: Preprocessed audio tensor, or None if loading fails.
        """
        audio = self.load_audio(audio_path)

        if audio is None:
            return None
        
        # Fix length (pad or truncate)
        audio = librosa.util.fix_length(
            audio, 
            size=self.max_samples,
            mode=self.pad_mode
        )

        # Normalize to [-1, 1] if needed
        if self.normalize:
            audio = librosa.util.normalize(audio)

        return torch.from_numpy(audio).float()
    

    def preprocess_batch(self, 
                         audio_paths: List[Union[str, bytes]]
                         ) -> List[Optional[torch.Tensor]]:
        """
        Preprocess a batch of audio files.

        Args:
            audio_paths (List[str or bytes]): List of paths to audio files.

        Returns:
            Tuple of (batched_audio_tensor, valid_indices)
        """
        batch_audio = [] 
        valid_indices = []

        for idx, path in enumerate(audio_paths):
            audio = self.preprocess_single(path)
            if audio is not None:
                batch_audio.append(audio)
                valid_indices.append(idx)
        
        if not batch_audio:
            logger.warning("No valid audio files found in batch.")
            return None, []
        
        # Stack into a single tensor
        return torch.stack(batch_audio), valid_indices
    
    def get_duration(self, audio_path: Union[str, bytes]) -> Optional[float]:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path (str or bytes): Path to the audio file.

        Returns:
            float: Duration in seconds, or None if loading fails.
        """
        try: 
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            logger.error(f"Error getting duration for {audio_path}: {e}")
            return None