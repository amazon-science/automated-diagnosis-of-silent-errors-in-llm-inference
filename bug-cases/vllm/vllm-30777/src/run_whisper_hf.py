import sys
import os
import json
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(_REPO_ROOT)
_ACTIVATION_DIR = os.environ.get("EKKA_ACTIVATION_DIR", os.path.join(_REPO_ROOT, "activations"))
from activation_collector.activation_collector import activation_collector
from scripts.generate_module_tree.get_model_tree import get_module_tree_and_flat_paths

model_id = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float32
dtype = torch.float16

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype).to(device)

# add activation collector
enable_activation_collector = True
if enable_activation_collector:
    activation_path = os.path.join(_ACTIVATION_DIR, "bug-cases/vllm/vllm-30777")
    # save_prefix = "hf_activation_fp32_transformer"
    # save_prefix = "hf_activation_fp32"
    save_prefix = "hf_activation"
    os.makedirs(activation_path, exist_ok=True)
    
    rank = int(os.environ.get("RANK", "0"))
    activation_collector.set_rank(rank)
    compressed, flat_paths = get_module_tree_and_flat_paths(model)  
    
    if activation_collector.get_rank() == 0:
        activation_save_path = f"{activation_path}/model_tree"
        os.makedirs(activation_save_path, exist_ok=True)
        with open(f"{activation_save_path}/model_tree_hf.json", "w") as f:
            json.dump(compressed, f, indent=2)
    
    hook_fns = [[None, None] for name in flat_paths]
    hook_modes = ["both"] * len(flat_paths)
    
    activation_collector.register_hooks(
        model=model,
        target_modules=flat_paths,
        hook_fns=hook_fns,
        hook_modes=hook_modes,
        save_path=f"{activation_path}/{save_prefix}",
        framework="hf"
    )
    print(f"Activation collector enabled. Saving to {activation_path}")

audio, sr = librosa.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BAC009S0038W0490.wav"), sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
if "input_features" in inputs:
    inputs["input_features"] = inputs["input_features"].to(dtype)

# Make decoding deterministic like your vLLM SamplingParams(temperature=0)
gen_kwargs = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1,
    # Optional but helps stability for Chinese:
    language="zh",
    task="transcribe",
)

with torch.no_grad():
    ids = model.generate(**inputs, **gen_kwargs)

text = processor.batch_decode(ids, skip_special_tokens=True)[0]

if enable_activation_collector:
    activation_collector.save_all()
    
print(text)
