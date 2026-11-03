import soundfile as sf
import torch
from vllm import LLM, SamplingParams
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ACTIVATION_DIR = os.environ.get("EKKA_ACTIVATION_DIR", os.path.join(_REPO_ROOT, "activations"))

# activation collector
enable_activation_collector = True
if enable_activation_collector:
    activation_path = os.path.join(_ACTIVATION_DIR, "bug-cases/vllm/vllm-30777")
    os.makedirs(activation_path, exist_ok=True)
    os.environ["VLLM_ACTIVATION_SAVE_PATH"] = activation_path

# Load your own audio
audio, sr = sf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BAC009S0038W0490.wav"))   # ← your file

llm = LLM(
    model="openai/whisper-large-v3-turbo",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    # ok
    # dtype=torch.float32,
    # buggy
    dtype=torch.float16
    
)

prompts = [
    {
        "prompt": "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": (audio, sr),     # ← only change
        },
    }
]

outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0, max_tokens=1),
)

print(outputs[0].outputs[0].text)
