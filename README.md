# Audio-LLMs

A prototype speech recognition model that connects a speech encoder to a text-only language model. A frozen Whisper encoder turns audio into features, a small trainable connector maps them into the embedding space of a frozen LLM, and the LLM generates the transcript.

## How it works

```
waveform (16 kHz, padded or cut to 30 s)
  -> WhisperFeatureExtractor (log-mel spectrogram)
  -> Whisper encoder, openai/whisper-small (frozen)
  -> AudioConnector (trainable): Conv1d, 4x shorter in time -> MLP -> LayerNorm
  -> Qwen/Qwen2.5-7B (frozen, loaded in 8-bit), fed as inputs_embeds
  -> transcript (greedy decoding)
```

- **Data.** `get_dataset.py` downloads the English (`en_us`) subset of FLEURS (`google/fleurs`) from Hugging Face, resamples it to 16 kHz, and writes `data/fleurs/{train,validation,test}.csv`.
- **Audio.** `AudioProcessor` loads mono audio with librosa at 16 kHz, returns `None` for clips shorter than 0.1 s, pads or truncates to 30 s, and peak-normalizes.
- **Dataset.** `AudioDataset` reads a CSV and returns each waveform with its transcript, tokenized by the tokenizer passed in (padded or truncated to 512 tokens).
- **Connector.** `AudioConnector` compresses the encoder output in time with a strided `Conv1d` (kernel and stride equal to `compression_factor`, default 4), then projects it with `Linear -> GELU -> Dropout -> Linear -> LayerNorm` to the LLM hidden size. `MultiModalASR` sets the input and output sizes from the two model configs.
- **Model.** `MultiModalASR` loads both pretrained models and freezes them by default, so only the connector is trained. It also enables gradient checkpointing on the LLM. `generate()` decodes greedily and returns a list of strings.

## Repository layout

| Path | Contents |
| --- | --- |
| [data_module/get_dataset.py](data_module/get_dataset.py) | Download FLEURS `en_us` and write one CSV per split |
| [data_module/audio_processor.py](data_module/audio_processor.py) | `AudioProcessor`: load, resample, pad or trim, normalize |
| [data_module/dataset.py](data_module/dataset.py) | `AudioDataset` (CSV to waveform plus tokens) and `collate_fn` |
| [models/audio_connector.py](models/audio_connector.py) | `AudioConnector`: encoder features to LLM embeddings |
| [models/multimodel_asr.py](models/multimodel_asr.py) | `MultiModalASR`: encoder, connector, and LLM with `forward` and `generate` |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Dependencies, managed with uv (Python 3.10) |

## Setup

```bash
git clone https://github.com/karthikbharadhwajKB/Audio-LLMs.git
cd Audio-LLMs
uv sync
```

On platforms other than Linux, `pyproject.toml` installs torch from the PyTorch CUDA 12.6 wheel index. Loading the LLM in 8-bit (the default) also needs `bitsandbytes` and `accelerate`, which are not listed in `pyproject.toml`:

```bash
uv pip install bitsandbytes accelerate
```

## Usage

There is no training or evaluation script yet. Run commands from the repository root, since the modules import each other as `data_module.*` and `models.*`.

Download FLEURS English and write the CSVs to `data/fleurs/` (the `data/` folder is git-ignored):

```bash
uv run python data_module/get_dataset.py
```

The audio processor and the connector can be used on their own:

```python
import torch
from data_module.audio_processor import AudioProcessor
from models.audio_connector import AudioConnector

processor = AudioProcessor(sample_rate=16_000, max_duration=30.0)
waveform = processor.preprocess_single("clip.wav")   # tensor of 480000 samples, or None

connector = AudioConnector(whisper_dim=768, llm_dim=4096, compression_factor=4)
features = torch.randn(1, 1500, 768)                 # Whisper-small encoder output for 30 s
embeddings = connector(features)                     # shape (1, 375, 4096)
```

## Status

Early prototype, August 2025. The data pipeline and model classes are written, but the full model has not been trained or run end to end. Known gaps in the current code:

- `MultiModalASR.__init__` calls `self.log_model_info()`, but the method is named `_log_model_info`, so building the model fails after both pretrained models have loaded.
- `forward()` passes only the audio embeddings to the LLM as `inputs_embeds`, together with the text `labels`. The transcript embeddings are not appended yet, so the lengths do not match (375 audio positions against 512 label tokens with the defaults) and the loss computation raises an error.
- `get_dataset.py` names the audio column `audio_file`, while `AudioDataset` expects `audio_path` by default (pass `audio_column="audio_file"`).
- `collate_fn` reads `item["transcription"]`, but dataset items store the text under `texts`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
