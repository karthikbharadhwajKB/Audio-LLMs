import torch 
import torch.nn as nn
from typing import Optional, List, Dict, Any
from transformers import (
    WhisperModel, WhisperFeatureExtractor,
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import logging 

# Internal import
from models.audio_connector import AudioConnector

# Set up logging
logger = logging.getLogger(__name__)


class MultiModalASR(nn.Module): 
    """
    Multi-modal ASR model integrating an audio encoder with a language model.

    Architecture:
        - Audio Encoder (e.g., Whisper) for audio feature extraction. (Frozen)
        - Trainable Audio Connector for modality bridging. (Trainable)
        - Language Model (e.g., LLaMA) for text generation. (Frozen)
    """
    def __init__(
            self, 
            audio_encoder_name: str = "openai/whisper-small",
            llm_model_name: str = "Qwen/Qwen2.5-7B",
            load_in_8bit: bool = True,
            compression_factor: int = 4,
            freeze_audio_encoder: bool = True,
            freeze_language_model: bool = True,
            enable_gradient_checkpointing: bool = True, # Enable gradient checkpointing for memory efficiency
    ): 
        """
        Initializes the MultiModalASR model.

        Args:
            audio_encoder_name (str): Pretrained Whisper model name.
            llm_model_name (str): Pretrained language model name.
            load_in_8bit (bool): Whether to load the LLM in 8-bit precision.
            compression_factor (int): Temporal compression factor for the audio connector.
            freeze_audio_encoder (bool): Whether to freeze the audio encoder parameters.
            freeze_language_model (bool): Whether to freeze the language model parameters.
            enable_gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency. 
        """
        # initialize the nn.Module
        super().__init__()

        self.audio_encoder_name = audio_encoder_name
        self.llm_model_name = llm_model_name
        self.compression_factor = compression_factor

        # Load the audio encoder (Whisper)
        logger.info(f"Loading audio encoder: {audio_encoder_name}")
        if self.audio_encoder_name.startswith("openai/whisper"):
            self.audio_encoder = WhisperModel.from_pretrained(audio_encoder_name)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_encoder_name)
        else:
            raise ValueError(f"Unsupported audio encoder: {audio_encoder_name}")

        # Load the LLM model
        logger.info(f"Loading language model: {llm_model_name}")
        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if load_in_8bit else torch.float32,
            device_map="auto" if load_in_8bit else None,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)


        # Set pad token if not present
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize the audio connector
        self.audio_connector = AudioConnector(
            whisper_dim=self.audio_encoder.config.d_model,
            llm_dim=self.llm.config.hidden_size,
            compression_factor=compression_factor,
        )

        # Freeze Models as specified
        if freeze_audio_encoder:
            self._freeze_audio_encoder()
        if freeze_language_model:
            self._freeze_llm()


        # Enable gradient checkpointing for memory efficiency
        if enable_gradient_checkpointing:
            if hasattr(self.llm, "gradient_checkpointing_enable"):
                self.llm.gradient_checkpointing_enable()


        logger.info("MultiModalASR model initialized.")
        self.log_model_info()


    def _freeze_audio_encoder(self):
        """Freezes the audio encoder parameters."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        logger.info("Audio encoder parameters frozen.")

    def _freeze_llm(self):
        """Freezes the LLM parameters."""
        for param in self.llm.parameters():
            param.requires_grad = False
        logger.info("Language model parameters frozen.")


    def _log_model_info(self): 
        """ Logging model information for debugging purposes. """
        audio_encoder_params = sum(p.numel() for p in self.audio_encoder.parameters())
        llm_params = sum(p.numel() for p in self.llm.parameters())
        connector_params = sum(p.numel() for p in self.audio_connector.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = audio_encoder_params + llm_params + connector_params

        logging.info("Model Parameters:")
        logging.info(f" Audio Encoder Parameters: {audio_encoder_params:,}")
        logging.info(f" LLM Parameters: {llm_params:,}")
        logging.info(f" Audio Connector Parameters: {connector_params:,}")
        logging.info(f" Trainable Parameters: {trainable_params:,}")
        logging.info(f" Total Parameters: {total_params:,} - ({trainable_params / total_params * 100:.2f}% trainable)")


    @torch.no_grad()
    def extract_audio_features(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from `Raw Audio` using the audio encoder.

        Pipeline: 
            Raw Audio -> Log-Mel Spectrogram -> Audio Encoder -> Audio Features

        Args:
            audio_inputs (torch.Tensor): Raw audio input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Extracted audio features of shape (batch_size, time_steps, feature_dim).
        """
        # Convert raw audio to Log-Mel spectrogram
        if audio_inputs.dim() == 2: # (batch_size, sequence_length) - batch of samples
            batch_size = audio_inputs.shape[0]
            audio_list = [audio_inputs[i].cpu().numpy() for i in range(batch_size)]
        else: # (sequence_length,) - single sample
            audio_list = [audio_inputs.cpu().numpy()]

        # Use feature extractor to get log-mel spectrogram
        mel_features = self.feature_extractor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(audio_inputs.device)

        # Extract features using the audio encoder
        encoder_outputs = self.audio_encoder.encoder(mel_features)

        return encoder_outputs.last_hidden_state  # (batch_size, time_steps, feature_dim) 


    def forward(
        self, 
        audio: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]: 
        """
        Foward pass for the MultiModalASR model training.

        Args: 
            audio (torch.Tensor): Raw audio input tensor of shape (batch_size, sequence_length).
            labels (torch.Tensor, optional): Tokenized text labels for computing loss. Shape (batch_size, seq_length).
            **kwargs: Additional arguments for future extensions.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits': Language model output logits of shape (batch_size, seq_length, vocab_size).
                - 'loss': Cross-entropy loss if labels are provided.
        """
        # Extract Audio Features
        audio_features = self.extract_audio_features(audio)

        # Pass through Audio Connector
        # Embeddings to match LLM input dimension
        connector_outputs = self.audio_connector(audio_features)

        # Move embeddings (connector outputs) to the same device as LLM
        connector_outputs = connector_outputs.to(self.llm.device)

        # labels processing
        if labels is not None: 
            labels = labels.to(self.llm.device)

        # Forward pass through the LLM
        outputs = self.llm(
            inputs_embeds = connector_outputs,
            labels = labels,
            return_dict = True,
            **kwargs
        )

        return outputs  

    @torch.no_grad()
    def generate(
        self, 
        audio: torch.Tensor,
        max_new_tokens: int = 256,
        **generate_kwargs: Any
    ) -> List[str]: 
        """
        Generate transcriptions from audio input.

        Args:
            audio (torch.Tensor): Raw audio input tensor of shape (batch_size, sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            **generate_kwargs: Additional generation parameters.

        Returns:
            List[str]: List of generated transcription strings.
        """ 
        self.eval()  # Set model to evaluation mode

        # Handle single sample input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # (1, sequence_length)

        # Extract Audio Features
        audio_features = self.extract_audio_features(audio)

        # Pass through Audio Connector
        connector_outputs = self.audio_connector(audio_features)
        # Move embeddings (connector outputs) to the same device as LLM
        connector_outputs = connector_outputs.to(self.llm.device)

        # Set default generation parameters
        default_generate_kwargs = {
           "max_new_tokens": max_new_tokens,
           "do_sample": False, # Greedy decoding for ASR
           "pad_token_id": self.tokenizer.pad_token_id,
           "eos_token_id": self.tokenizer.eos_token_id,
           **generate_kwargs
        }  

        # Generate tokens using the LLM
        generated_ids = self.llm.generate(
            inputs_embeds = connector_outputs,
            **default_generate_kwargs
        )

        # Decode generated token IDs to strings
        transcriptions = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
        )

        return transcriptions
        


    