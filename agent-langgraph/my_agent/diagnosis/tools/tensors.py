"""LLM tools for inspecting the activation tensors registered for an alignment trial."""

from typing import List, Dict, Any, Optional, Tuple
from langchain.tools import tool
import math
import torch
import torch.distributed.tensor
import logging

from my_agent.diagnosis.runtime import get_tensor, ACTIVATION_STORE


@tool("get_tensor_pair_sum")
def get_tensor_pair_sum(keyA: str, keyB: str) -> dict:
    """Take two tensors A and B, and return their total sums.
    
    Args:
        keyA (str): The key of the first tensor.
        keyB (str): The key of the second tensor.

    Returns:
        dict: A dictionary with the following keys:
            "sum_A": float,      # sum of the first tensor
            "sum_B": float       # sum of the second tensor
    """
    logging.info(f"[Tool][get_tensor_pair_sum] Getting tensor pair sum for {keyA} and {keyB}")
    total_A = get_tensor(keyA).sum().item()
    total_B = get_tensor(keyB).sum().item()
    return {"sum_A": total_A, "sum_B": total_B}


def _rand_multi_indices(
    shape: Tuple[int, ...],
    k: int,
    seed: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Random k indices over the flattened space, uniformly without replacement."""
    numel = int(math.prod(shape))
    if numel == 0 or k <= 0:
        return []
    k = min(k, numel)

    g = torch.Generator()
    if seed is not None:
        g = g.manual_seed(int(seed))
    flat = torch.randperm(numel, generator=g, dtype=torch.long)[:k]  # [k]
    coords = torch.unravel_index(flat, shape)  # tuple[tensor[k], ...]
    return [tuple(int(coords[d][i]) for d in range(len(shape))) for i in range(k)]


# def _rand_multi_indices(
#     shape: Tuple[int, ...],
#     k: int,
#     seed: Optional[int] = None,
#     include_first_n_lastdim: int = 0,
# ) -> List[Tuple[int, ...]]:
#     import math, torch
#     numel = int(math.prod(shape))
#     if numel == 0:
#         return []

#     # deterministic: first N along last dim with zero prefix
#     det: List[Tuple[int, ...]] = []
#     if include_first_n_lastdim > 0:
#         n = min(max(include_first_n_lastdim, 0), shape[-1])
#         prefix = (0,) * (len(shape) - 1)
#         det = [(*prefix, i) for i in range(n)]

#     if k <= 0:
#         return det

#     k = min(k, numel)
#     g = torch.Generator()
#     if seed is not None:
#         g = g.manual_seed(int(seed))

#     flat = torch.randperm(numel, generator=g, dtype=torch.long)[:k]
#     coords = torch.unravel_index(flat, shape)
#     rand_list = [tuple(int(coords[d][i]) for d in range(len(shape))) for i in range(k)]

#     # merge & dedup
#     seen, merged = set(), []
#     for idx in det + rand_list:
#         if idx not in seen:
#             seen.add(idx)
#             merged.append(idx)
#     return merged
@tool("infer_tensor_match")
def infer_tensor_match(
    keyA: str,
    keyB: str,
    sample_size: int = 10,
    threshold: float = 1e-3,
    seed: Optional[int] = None,
    max_hits_per_sample: int = 5,
    include_first_n_lastdim: int = 5,
    include_last_n_lastdim: int = 5,
    include_max_along_lastdim: bool = True,
    compare_only_last_dim_slice: bool = True,    
) -> Dict[str, Any]:
    """
    Sample a few random elements from tensor A, find indices in tensor B whose values
    differ by <= threshold, and return matches. No dimension mapping inference is done.

    Args:
        keyA (str): The key of the first tensor.
        keyB (str): The key of the second tensor.
        sample_size (int, optional): Number of random elements to sample from A.
            Default = 10. Clipped to [1, A.numel()].
        threshold (float, optional): Absolute difference tolerance for considering
            values in B as matches. Default = 1e-3.
        seed (int, optional): Random seed for reproducible sampling. Default = None
            (nondeterministic).
        max_hits_per_sample (int, optional): Maximum number of matches to keep in B
            for each sampled element from A. Default = 5.
        include_first_n_lastdim (int, optional): Include the first N indices along
            the last dimension of A as deterministic samples. Default = 5.
        include_last_n_lastdim (int, optional): Include the last N indices along
            the last dimension of A as deterministic samples. Default = 5.
        include_max_along_lastdim (bool, optional): Whether to include the index of
            the maximum value along the last dimension of A (with zero prefix) as a
            deterministic sample. Default = True.
        compare_only_last_dim_slice (bool, optional): Whether to compare only the first/last N indices along
            the last dimension of A and B. Default = True.

    Returns:
        dict: A dictionary with the following keys:
            {
                "shape_A": list[int],      # shape of A
                "shape_B": list[int],      # shape of B
                "samples_used": int,       # number of sampled indices from A
                "sample_categories": dict, # category name -> list of indices
                "matches": [               # list of match dictionaries
                    {
                        "category": str,
                        "a_index": [...],  # coordinates in A
                        "b_index": [...],  # coordinates in B
                        "a_value": float,  # value from A
                        "b_value": float   # matching value from B
                    },
                    ...
                ],
                "notes": str               # notes if empty, mismatched sizes, or no matches
            }
    """
    logging.info(f"[Tool][infer_tensor_match] Comparing tensor pair {keyA} and {keyB}, args: {locals()}")
    A = get_tensor(keyA).detach().cpu()
    B = get_tensor(keyB).detach().cpu()
    shapeA = tuple(A.shape)
    shapeB = tuple(B.shape)

    if A.numel() == 0 or B.numel() == 0:
        return {
            "shape_A": list(shapeA),
            "shape_B": list(shapeB),
            "samples_used": 0,
            "matches": [],
            "notes": "Empty tensor(s)",
            "sample_categories": {},
        }

    if A.numel() != B.numel():
        return {
            "shape_A": list(shapeA),
            "shape_B": list(shapeB),
            "samples_used": 0,
            "matches": [],
            "notes": "Number of elements in A and B do not match",
            "sample_categories": {},
        }

    D_A = shapeA[-1]
    D_B = shapeB[-1]
    if D_A != D_B:
        return {
            "shape_A": list(shapeA),
            "shape_B": list(shapeB),
            "samples_used": 0,
            "matches": [],
            "notes": "Last dimensions do not match",
            "sample_categories": {},
        }

    if compare_only_last_dim_slice:
        if shapeA[-1] != shapeB[-1]:
            return {
                "shape_A": list(shapeA),
                "shape_B": list(shapeB),
                "samples_used": 0,
                "matches": [],
                "notes": "Last dimensions do not match for compare_only_last_dim_slice",
                "sample_categories": {},
            }

        D = shapeA[-1]
        first_n = min(include_first_n_lastdim, D)
        last_n = min(include_last_n_lastdim, D)

        # Prefix: (0, ..., 0) to slice the last dimension
        a_prefix = (0,) * (A.ndim - 1)
        b_prefix = (0,) * (B.ndim - 1)

        # Sample a sample size of indices along the last dimension
        g = torch.Generator()
        stride_indices = torch.randperm(D, generator=g)[:sample_size]
        cats = {
            "first_n_lastdim": [a_prefix + (i,) for i in range(first_n)],
            "last_n_lastdim": [a_prefix + (i,) for i in range(D - last_n, D)],
            "stride_lastdim": [a_prefix + (i,) for i in stride_indices
                                if i >= first_n and i < D - last_n],  # avoid overlap
        }

        matches = []
        b_slice = B[b_prefix + (slice(None),)]  # shape: [D]

        for cat, a_indices in cats.items():
            for a_idx in a_indices:
                i = a_idx[-1]
                a_val = A[a_idx]
                diffs = (b_slice - a_val).abs()
                nz = (diffs <= threshold).nonzero(as_tuple=False)
                for j in nz[:max_hits_per_sample].flatten():
                    b_idx = b_prefix + (int(j.item()),)
                    matches.append({
                        "category": cat,
                        "a_index": list(a_idx),
                        "b_index": list(b_idx),
                        "a_value": float(a_val.item()),
                        "b_value": float(B[b_idx].item()),
                    })
        result = {
            "shape_A": list(shapeA),
            "shape_B": list(shapeB),
            "samples_used": sum(len(v) for v in cats.values()),
            "sample_categories": {k: [list(x) for x in v] for k, v in cats.items()},
            "matches": matches,
            "notes": "No matches within threshold." if not matches else "",
        }
        logging.info(f"[Tool][infer_tensor_match] {result}")
        return result


    # === Regular full match mode (includes random + max sampling) ===
    from typing import List, Tuple, Dict, Any
    def _rand_multi_indices(shape: Tuple[int, ...], k: int, seed: Optional[int] = None) -> List[Tuple[int, ...]]:
        import torch
        import math
        numel = int(math.prod(shape))
        if numel == 0 or k <= 0:
            return []
        k = min(k, numel)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        flat = torch.randperm(numel, generator=g)[:k]
        coords = torch.unravel_index(flat, shape)
        return [tuple(int(coords[d][i]) for d in range(len(shape))) for i in range(k)]

    zero_prefix = (0,) * (A.ndim - 1)
    D = shapeA[-1]

    cats: Dict[str, List[Tuple[int, ...]]] = {
        "first_n_lastdim": [],
        "last_n_lastdim": [],
        "max_value_lastdim": [],
        "random": [],
    }

    if include_first_n_lastdim > 0:
        cats["first_n_lastdim"] = [zero_prefix + (i,) for i in range(min(include_first_n_lastdim, D))]

    if include_last_n_lastdim > 0:
        cats["last_n_lastdim"] = [zero_prefix + (i,) for i in range(D - min(include_last_n_lastdim, D), D)]

    if include_max_along_lastdim:
        max_i = int(torch.argmax(A[zero_prefix + (slice(None),)]).item())
        cats["max_value_lastdim"] = [zero_prefix + (max_i,)]

    cats["random"] = _rand_multi_indices(shapeA, sample_size, seed)

    merged_indices = [(cat, idx) for cat, indices in cats.items() for idx in indices]

    matches: List[Dict[str, Any]] = []
    for cname, a_idx in merged_indices:
        a_val = A[a_idx]
        diffs = (B - a_val).abs()
        nz = (diffs <= threshold).nonzero(as_tuple=False)
        for row in nz[:max_hits_per_sample]:
            b_idx = tuple(int(i) for i in row.tolist())
            matches.append({
                "category": cname,
                "a_index": list(a_idx),
                "b_index": list(b_idx),
                "a_value": float(a_val.item()),
                "b_value": float(B[b_idx].item()),
            })

    return {
        "shape_A": list(shapeA),
        "shape_B": list(shapeB),
        "samples_used": len(merged_indices),
        "sample_categories": {k: [list(x) for x in v] for k, v in cats.items()},
        "matches": matches,
        "notes": "No matches within threshold." if not matches else "",
    }


@tool("list_tensor_keys")
def list_tensor_keys() -> List[str]:
    """Return a list of all available tensor keys for referencing tensors (e.g., used as arguments for infer_tensor_match and get_tensor_pair_sum).
    
    Keys are in the form derived from torch.load
    
    Key format:
        Inputs: 
            [hf/vllm]_inputs_[seq_idx]_input_[input_idx]
            [hf/vllm]_inputs_[seq_idx]_kwargs_[kwargs_name]
        Outputs:
            [hf/vllm]_outputs_[seq_idx]_[output_idx]

    Args:
        None
    """
    logging.info(f"[Tool][list_tensor_keys] Listing tensor keys")
    keys = list(ACTIVATION_STORE.keys())
    # print(keys)
    return keys
