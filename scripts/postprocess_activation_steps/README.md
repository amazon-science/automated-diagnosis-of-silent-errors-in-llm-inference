# postprocess_activation_steps

Extracts the activations of a single decode step from activations collected over
several generation steps, so that reference and target activations can be compared
step by step. Run it once per activation prefix (for example the target engine's
prefix, then the reference bf16 and fp32 prefixes), each time with the call sequence
recorded for that framework.

```bash
python3 scripts/postprocess_activation_steps/postprocess_activation_steps.py \
  --input-dir     "$EKKA_ACTIVATION_DIR/bug-cases/sglang/sglang-10344_8tokens" \
  --call-sequence bug-cases/sglang/sglang-10344/call_sequence_sglang/call_sequence_rank0.log \
  --output-root   "$EKKA_ACTIVATION_DIR/bug-cases/sglang/sglang-10344_decode_1" \
  --step-idx 1 \
  --prefix sglang_activation_rank0
```
