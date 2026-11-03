# Original Issue

By continuously increasing the tp, I found that the test performance deteriorated as tp grew larger (since I have 4 cards, the maximum setting for tp was 4). Additionally, when testing with Qwen3-reranker-0.6B, I noticed that when tp was set to 4, the memory usage of one card could reach around 10G, and the total for 4 cards was close to 40G! I'm not quite sure what caused these two abnormal results.

# Reproducible Example

The issue can be reproduced by running the following counter-example where the HuggingFace and vLLM output under the same random seed and configuration are significantly different.

```
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    queries = [
        "What is the capital of China?",
        # "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        # "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    model = get_model()
    outputs = model.score(queries, documents)
```

## Result
HF TP=4 scores
scores: [0.99951171875, 0.99951171875]

vLLM TP=4 scores
scores [0.3294568359851837, 0.25639307498931885]
