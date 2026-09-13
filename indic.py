import torch
import numpy as np
import soundfile as sf
import os
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Workaround for torchaudio 2.11 on Windows: torchaudio.load() requires
# torchcodec + FFmpeg shared libs, which fail to load (libtorchcodec DLLs).
# Patch torchaudio.load to use soundfile (already a dependency) instead.
# Only affects local ref-audio loading; Resample transform is pure torch.
import torchaudio


def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, **kwargs):
    data, sr = sf.read(str(uri), always_2d=True)  # (time, channel) float64
    if frame_offset > 0:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    wav = torch.from_numpy(data.T).to(torch.float32)  # (channel, time)
    if not channels_first:
        wav = wav.T
    return wav, sr


torchaudio.load = _sf_load

# Import required modules from f5-tts
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
    get_tokenizer,
)

# Load IndicF5 from Hugging Face
# Note: The HF model's model.py has a bug (missing ckpt_path in load_model call).
# We bypass it by downloading model files directly and constructing the model.
repo_id = "ai4bharat/IndicF5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download required files
print("Downloading vocab.txt...")
vocab_path = hf_hub_download(repo_id, filename="checkpoints/vocab.txt")
print(f"Vocab path: {vocab_path}")

print("Downloading model.safetensors...")
model_path = hf_hub_download(repo_id, filename="model.safetensors")
print(f"Model path: {model_path}")

# Initialize tokenizer
tokenizer = "custom"
vocab_char_map, vocab_size = get_tokenizer(vocab_path, tokenizer)
print(f"Vocab size: {vocab_size}")

# Create model architecture
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

print("Creating model architecture...")
model = CFM(
    transformer=DiT(
        **model_cfg,
        text_num_embeds=vocab_size,
        mel_dim=100,
    ),
    mel_spec_kwargs=dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    ),
    odeint_kwargs=dict(method="euler"),
    vocab_char_map=vocab_char_map,
).to(device)

# Load checkpoint from safetensors
print("Loading model weights from safetensors...")
from f5_tts.infer.utils_infer import load_checkpoint

checkpoint = load_file(model_path, device=str(device))

# Handle EMA model state dict format
if "ema_model_state_dict" not in checkpoint:
    checkpoint = {"ema_model_state_dict": checkpoint}

# Extract model state dict (remove ema_model. and _orig_mod. prefixes)
model_state_dict = {
    k.replace("ema_model.", "").replace("_orig_mod.", ""): v
    for k, v in checkpoint["ema_model_state_dict"].items()
    if k not in ["initted", "step"]
}

model.load_state_dict(model_state_dict, strict=False)
print("Model loaded successfully!")

# Load vocoder
print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Wrapper class that mimics the original model API
class IndicF5Wrapper:
    def __init__(self, model, vocoder, device):
        self.model = model
        self.vocoder = vocoder
        self.device = device

    def __call__(self, text, ref_audio_path, ref_text):
        """
        Generate speech given a reference audio & text input.

        Args:
            text (str): The text to be synthesized.
            ref_audio_path (str): Path to the reference audio file.
            ref_text (str): The reference text.

        Returns:
            np.array: Generated waveform.
        """
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file {ref_audio_path} not found.")

        # Load reference audio & text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

        self.model.to(self.device)
        self.vocoder.to(self.device)

        # Perform inference
        audio, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            self.model,
            self.vocoder,
            mel_spec_type="vocos",
            speed=1.0,
            device=self.device,
        )

        return audio

# Create model instance
model = IndicF5Wrapper(model, vocoder, device)

# Generate speech
print("Generating speech...")
audio = model(
    "Hello, ମୁଁ ସୌରଜିତ ସ୍ୱାଗତ କରୁଛି ମୋର YouTube channel କୁ। ଆଜି ଆମେ Artificial Intelligence ବିଷୟରେ ପଢିବା।",
    ref_audio_path="prompts/record_out.wav",
    ref_text="Hello, ମୁଁ ସୌରଜିତ ସ୍ୱାଗତ କରୁଛି ମୋର YouTube channel କୁ। ଆଜି ଆମେ Artificial Intelligence ବିଷୟରେ ପଢିବା।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved succesfully.")