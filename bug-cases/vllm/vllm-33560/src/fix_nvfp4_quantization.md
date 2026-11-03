# Fix for NVFP4 Quantization KeyError: 'weight_scale'

## Problem Description

When running NVFP4 quantized models (e.g., `RedHatAI/Qwen3-8B-NVFP4`) with `compressed-tensors` library version 0.13.0, the following error occurs:

```
KeyError: 'weight_scale'
  File "compressed_tensors/compressors/quantized_compressors/fp4_quantized.py", line 126, in decompress_weight
    scale = compressed_data["weight_scale"]
            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
```

## Root Cause

1. **Missing Buffers**: The `compressed-tensors` library's `decompress_module()` method only collects parameters via `named_parameters()`, but the `weight_scale` tensors are stored as buffers, not parameters.

2. **Transformers Filtering**: When loading NVFP4 quantized models, `transformers` filters out the `weight_scale` tensors during model initialization, showing warnings like:
   ```
   Some weights of the model checkpoint at RedHatAI/Qwen3-8B-NVFP4 were not used when initializing...
   ['model.layers.0.mlp.down_proj.weight_scale', ...]
   ```

3. **Dtype Mismatch**: Even after loading the scales, there are dtype compatibility issues between float8 scales and the model's inference dtype (float16).

## Solution

Three files need to be patched:

---

## 1. Fix for `compressed_tensors/compressors/base.py`

**File:** `<site-packages>/compressed_tensors/compressors/base.py`

**Location:** Around line 183-190

```diff
         quantization_args = quantization_scheme.weights
         compressed_data = {}
         for name, parameter in module.named_parameters():
             compressed_data[name] = parameter
+        # Also include buffers (e.g., scales) which may not be parameters
+        for name, buffer in module.named_buffers():
+            compressed_data[name] = buffer

         return self.decompress_weight(
             compressed_data=compressed_data, quantization_args=quantization_args
         ).to(device)
```

**Explanation:** This ensures that buffers (including `weight_scale`) are included in the `compressed_data` dictionary passed to `decompress_weight()`.

---

## 2. Fix for `driver_hf_v2_single_gpu.py`

**File:** `<repo>/drivers/driver_hf_v2_single_gpu.py`

**Location:** After line 88, before `load_hf_model_and_tokenizer()`

```diff
 # ---------- model / gen ----------

+def fix_missing_quantization_buffers(model, model_name, logger):
+    """
+    Manually load missing quantization scale tensors that transformers doesn't load.
+    This is needed for NVFP4 quantized models where weight_scale buffers are filtered out.
+    """
+    try:
+        from safetensors import safe_open
+        from huggingface_hub import hf_hub_download
+        import json
+
+        # Download the index to find which files contain the scales
+        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
+        with open(index_path) as f:
+            index = json.load(f)
+
+        weight_map = index.get("weight_map", {})
+
+        # Group missing scales by file
+        missing_scales = {}
+        for module_name, module in model.named_modules():
+            if hasattr(module, 'weight_packed') and not hasattr(module, 'weight_scale'):
+                scale_key = f"{module_name}.weight_scale"
+                if scale_key in weight_map:
+                    shard_file = weight_map[scale_key]
+                    if shard_file not in missing_scales:
+                        missing_scales[shard_file] = []
+                    missing_scales[shard_file].append((module_name, scale_key))
+
+        if not missing_scales:
+            return
+
+        logger.info(f"Fixing {sum(len(v) for v in missing_scales.values())} missing quantization scale buffers")
+
+        # Load and register the missing scales
+        for shard_file, scales_to_load in missing_scales.items():
+            shard_path = hf_hub_download(model_name, shard_file)
+            with safe_open(shard_path, framework='pt', device='cpu') as f:
+                for module_name, scale_key in scales_to_load:
+                    module = dict(model.named_modules())[module_name]
+                    scale_tensor = f.get_tensor(scale_key)
+                    # Convert float8 scales to float16 to match typical model dtype
+                    # Using float16 since most inference runs in float16
+                    if scale_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
+                        scale_tensor = scale_tensor.to(torch.float32).to(torch.float16)
+                    module.register_buffer('weight_scale', scale_tensor)
+
+        logger.info("Successfully loaded missing quantization buffers")
+    except Exception as e:
+        logger.warning(f"Could not fix missing quantization buffers: {e}")
+
+
 def load_hf_model_and_tokenizer(model_cfg: dict, logger: logging.Logger, mode: str = "generate"):
     model_name = model_cfg["name"]
```

**Location:** Inside `load_hf_model_and_tokenizer()`, after model loading

```diff
     if mode == "reranker":
         model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
     else:
         model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

+    # Fix missing quantization buffers for NVFP4 models
+    fix_missing_quantization_buffers(model, model_name, logger)
+
     model.eval()
     return model, tokenizer
```

**UPDATE (2026-03-06):** Function now includes **FP32 support**:

```diff
         logger.info(f"Fixing {sum(len(v) for v in missing_scales.values())} missing quantization scale buffers")

+        # Detect model's actual dtype from embed_tokens or first layer norm
+        model_dtype = torch.float16  # default
+        for name, param in model.named_parameters():
+            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
+                model_dtype = param.dtype
+                logger.info(f"Detected model dtype: {model_dtype} from parameter '{name}'")
+                break
+
         # Load and register the missing scales
         for shard_file, scales_to_load in missing_scales.items():
             shard_path = hf_hub_download(model_name, shard_file)
             with safe_open(shard_path, framework='pt', device='cpu') as f:
                 for module_name, scale_key in scales_to_load:
                     module = dict(model.named_modules())[module_name]
                     scale_tensor = f.get_tensor(scale_key)
-                    # Convert float8 scales to float16 to match typical model dtype
-                    # Using float16 since most inference runs in float16
+                    # Convert float8 scales to match model dtype
                     if scale_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
-                        scale_tensor = scale_tensor.to(torch.float32).to(torch.float16)
+                        scale_tensor = scale_tensor.to(torch.float32).to(model_dtype)
                     module.register_buffer('weight_scale', scale_tensor)

-        logger.info("Successfully loaded missing quantization buffers")
+        logger.info(f"Successfully loaded missing quantization buffers (dtype: {model_dtype})")
```

**Explanation:** This function manually loads the `weight_scale` tensors from the safetensors files that were filtered out by `transformers`, and registers them as buffers on the appropriate modules. It now auto-detects the model's dtype (FP32, FP16, or BF16) and converts scales accordingly.

---

## 3. Fix for `fp4_quantized.py`

**File:** `<site-packages>/compressed_tensors/compressors/quantized_compressors/fp4_quantized.py`

**Location:** Around line 119-134

```diff
     def decompress_weight(
         self,
         compressed_data: Dict[str, Tensor],
         quantization_args: Optional[QuantizationArgs] = None,
     ) -> torch.Tensor:
+        # Debug: print available keys
+        if "weight_scale" not in compressed_data:
+            print(f"[DEBUG] Missing weight_scale! Available keys: {list(compressed_data.keys())}")
+            raise KeyError(f"weight_scale not found. Available: {list(compressed_data.keys())}")
+
         weight = compressed_data["weight_packed"]
         global_scale = compressed_data["weight_global_scale"]
         scale = compressed_data["weight_scale"]
         m, n = weight.shape
-        # TODO: use a user provided dequant dtype
-        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
+        # Use the scale's dtype to determine decompression dtype
+        # This will be float16, bfloat16, or float32 depending on model config
+        dequant_dtype = scale.dtype
+        if dequant_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
+            # Fallback to float16 if scale has an unexpected dtype
+            dequant_dtype = torch.float16
+        unpacked = unpack_fp4_from_uint8(weight, m, n * 2, dtype=dequant_dtype)
         decompressed_weight = dequantize(
-            x_q=unpacked, scale=scale, global_scale=global_scale, dtype=unpacked.dtype
+            x_q=unpacked, scale=scale, global_scale=global_scale, dtype=dequant_dtype
         )

         return decompressed_weight
```

**Explanation:** This ensures that the unpacking and dequantization use a consistent dtype based on the scale tensor's dtype. Now supports FP32, FP16, and BF16 models, preventing dtype mismatch errors during matrix multiplication.

---

## Testing

After applying these patches, the model runs successfully:

```bash
python3 \
  <repo>/drivers/driver_hf_v2_single_gpu.py \
  --config <repo>/drivers/hf_config_qwen3_fp8.yaml
```

**Expected output:**
```
2026-03-06 07:41:47,635 [INFO] Fixing 252 missing quantization scale buffers
2026-03-06 07:41:48,249 [INFO] Successfully loaded missing quantization buffers
...
2026-03-06 07:45:58,595 [INFO] Generated:
Answer: The number of students in the band is 120 x 0.30 = <<120*0.30=36>>36.
2026-03-06 07:45:58,602 [INFO] ### HF run completed successfully.
```

---

## Environment

- **Python**: 3.12
- **compressed-tensors**: 0.13.0
- **transformers**: 4.56.1
- **torch**: 2.x (with CUDA support)
- **Model**: RedHatAI/Qwen3-8B-NVFP4 (NVFP4 quantized)

---

## Notes

1. The fix in `base.py` is the minimal library-level change needed and could be contributed upstream to the `compressed-tensors` repository.

2. The fix in `driver_hf_v2_single_gpu.py` is a workaround for the fact that `transformers` filters out the scale buffers during model initialization. This is model-loading code specific to your use case.

3. The fix in `fp4_quantized.py` ensures dtype consistency during decompression, which prevents runtime errors.

4. These changes are specific to NVFP4 quantized models and should not affect other quantization formats.

5. **UPDATE (2026-03-06)**: Now supports **FP32, FP16, and BF16** model dtypes. The code auto-detects the model's dtype and converts scales/weights accordingly. To use FP32:
   ```yaml
   model:
     name: "RedHatAI/Qwen3-8B-NVFP4"
     torch_dtype: "float32"
   ```

---

## Related Issues

- This issue is related to how `transformers` loads quantized models and filters out certain tensors
- Similar issues may occur with other FP4/FP8 quantization formats
- The `compressed-tensors` library may need updates to better handle these edge cases

---

**Date:** 2026-03-06
**Tested on:** RedHatAI/Qwen3-8B-NVFP4 model
