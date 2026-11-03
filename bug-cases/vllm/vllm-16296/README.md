# vLLM Issue 16296

## Run HF Example
torchrun --standalone --nproc-per-node 8 test_llama4_hf.py

### HF Output
[['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.'], ['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.'], ['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.'], ['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.'], ['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.']]


## Run vLLM Example
python3 test_llama4_vllm.py

### vLLM Buggy Output
[' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers', ' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers', ' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers', ' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers', ' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers']


### vLLM Fix Output
[' a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.', ' a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.', ' a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.', ' a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.', ' a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.']

## Buggy Version
vLLM v0.8.3

To run this version, please manually add the fix otherwise there will be index error. https://github.com/vllm-project/vllm/pull/16209/files

## Fix Version
vLLM v0.8.4

## 1B Sanity Check

First token logits (HF): tensor([ 7.2500,  6.0938,  7.3750,  ..., -0.1504, -0.1504, -0.1514],
       dtype=torch.bfloat16)
First token logits (vLLM): tensor([ 7.2500,  6.1250,  7.3750,  ..., -0.1553, -0.1553, -0.1553],
       dtype=torch.bfloat16)
Number of tokens compared: 50
Vocab size: 128256

=== Difference Summary ===
Mean L2 norm:     18.625000
Max L2 norm:      59.500000
Mean Max-Abs Diff:0.226562
Max Max-Abs Diff: 0.437500
Min Cosine Sim:   0.996094
Mean Cosine Sim:  1.000000

## Issue Link
https://github.com/vllm-project/vllm/issues/16296

## Fix Link
https://github.com/vllm-project/vllm/issues/16311