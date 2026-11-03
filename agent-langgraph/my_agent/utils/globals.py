def inverse_perm(idx_list):
    """Return inverse permutation of a list of indices."""
    inv = [0] * len(idx_list)
    for new_pos, old_pos in enumerate(idx_list):
        inv[old_pos] = new_pos
    return inv

# ERROR_THRESHOLD = 5
ERROR_THRESHOLD = 1.5
EPS_DENOMINATOR = 1e-3 # at the scale of BF16 error 
# derived from precison roundoff error of FP32 and BF16
EPS_FP32_BF16 = 0.0039 
EPS_BF16_BF16 = 0.0078
COMPONENT_MAPPING_MAX_RETRY = 3
COMPONENT_MAPPING_MAX_ERROR_FEEDBACK = 3
ACTIVATION_ALIGNMENT_MAX_RETRY = 5
ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK = 5


INTERLEAVE_MAP_Q = [0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95, 32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127, 128, 192, 129, 193, 130, 194, 131, 195, 132, 196, 133, 197, 134, 198, 135, 199, 136, 200, 137, 201, 138, 202, 139, 203, 140, 204, 141, 205, 142, 206, 143, 207, 144, 208, 145, 209, 146, 210, 147, 211, 148, 212, 149, 213, 150, 214, 151, 215, 152, 216, 153, 217, 154, 218, 155, 219, 156, 220, 157, 221, 158, 222, 159, 223, 160, 224, 161, 225, 162, 226, 163, 227, 164, 228, 165, 229, 166, 230, 167, 231, 168, 232, 169, 233, 170, 234, 171, 235, 172, 236, 173, 237, 174, 238, 175, 239, 176, 240, 177, 241, 178, 242, 179, 243, 180, 244, 181, 245, 182, 246, 183, 247, 184, 248, 185, 249, 186, 250, 187, 251, 188, 252, 189, 253, 190, 254, 191, 255, 256, 320, 257, 321, 258, 322, 259, 323, 260, 324, 261, 325, 262, 326, 263, 327, 264, 328, 265, 329, 266, 330, 267, 331, 268, 332, 269, 333, 270, 334, 271, 335, 272, 336, 273, 337, 274, 338, 275, 339, 276, 340, 277, 341, 278, 342, 279, 343, 280, 344, 281, 345, 282, 346, 283, 347, 284, 348, 285, 349, 286, 350, 287, 351, 288, 352, 289, 353, 290, 354, 291, 355, 292, 356, 293, 357, 294, 358, 295, 359, 296, 360, 297, 361, 298, 362, 299, 363, 300, 364, 301, 365, 302, 366, 303, 367, 304, 368, 305, 369, 306, 370, 307, 371, 308, 372, 309, 373, 310, 374, 311, 375, 312, 376, 313, 377, 314, 378, 315, 379, 316, 380, 317, 381, 318, 382, 319, 383, 384, 448, 385, 449, 386, 450, 387, 451, 388, 452, 389, 453, 390, 454, 391, 455, 392, 456, 393, 457, 394, 458, 395, 459, 396, 460, 397, 461, 398, 462, 399, 463, 400, 464, 401, 465, 402, 466, 403, 467, 404, 468, 405, 469, 406, 470, 407, 471, 408, 472, 409, 473, 410, 474, 411, 475, 412, 476, 413, 477, 414, 478, 415, 479, 416, 480, 417, 481, 418, 482, 419, 483, 420, 484, 421, 485, 422, 486, 423, 487, 424, 488, 425, 489, 426, 490, 427, 491, 428, 492, 429, 493, 430, 494, 431, 495, 432, 496, 433, 497, 434, 498, 435, 499, 436, 500, 437, 501, 438, 502, 439, 503, 440, 504, 441, 505, 442, 506, 443, 507, 444, 508, 445, 509, 446, 510, 447, 511, 512, 576, 513, 577, 514, 578, 515, 579, 516, 580, 517, 581, 518, 582, 519, 583, 520, 584, 521, 585, 522, 586, 523, 587, 524, 588, 525, 589, 526, 590, 527, 591, 528, 592, 529, 593, 530, 594, 531, 595, 532, 596, 533, 597, 534, 598, 535, 599, 536, 600, 537, 601, 538, 602, 539, 603, 540, 604, 541, 605, 542, 606, 543, 607, 544, 608, 545, 609, 546, 610, 547, 611, 548, 612, 549, 613, 550, 614, 551, 615, 552, 616, 553, 617, 554, 618, 555, 619, 556, 620, 557, 621, 558, 622, 559, 623, 560, 624, 561, 625, 562, 626, 563, 627, 564, 628, 565, 629, 566, 630, 567, 631, 568, 632, 569, 633, 570, 634, 571, 635, 572, 636, 573, 637, 574, 638, 575, 639, 640, 704, 641, 705, 642, 706, 643, 707, 644, 708, 645, 709, 646, 710, 647, 711, 648, 712, 649, 713, 650, 714, 651, 715, 652, 716, 653, 717, 654, 718, 655, 719, 656, 720, 657, 721, 658, 722, 659, 723, 660, 724, 661, 725, 662, 726, 663, 727, 664, 728, 665, 729, 666, 730, 667, 731, 668, 732, 669, 733, 670, 734, 671, 735, 672, 736, 673, 737, 674, 738, 675, 739, 676, 740, 677, 741, 678, 742, 679, 743, 680, 744, 681, 745, 682, 746, 683, 747, 684, 748, 685, 749, 686, 750, 687, 751, 688, 752, 689, 753, 690, 754, 691, 755, 692, 756, 693, 757, 694, 758, 695, 759, 696, 760, 697, 761, 698, 762, 699, 763, 700, 764, 701, 765, 702, 766, 703, 767, 768, 832, 769, 833, 770, 834, 771, 835, 772, 836, 773, 837, 774, 838, 775, 839, 776, 840, 777, 841, 778, 842, 779, 843, 780, 844, 781, 845, 782, 846, 783, 847, 784, 848, 785, 849, 786, 850, 787, 851, 788, 852, 789, 853, 790, 854, 791, 855, 792, 856, 793, 857, 794, 858, 795, 859, 796, 860, 797, 861, 798, 862, 799, 863, 800, 864, 801, 865, 802, 866, 803, 867, 804, 868, 805, 869, 806, 870, 807, 871, 808, 872, 809, 873, 810, 874, 811, 875, 812, 876, 813, 877, 814, 878, 815, 879, 816, 880, 817, 881, 818, 882, 819, 883, 820, 884, 821, 885, 822, 886, 823, 887, 824, 888, 825, 889, 826, 890, 827, 891, 828, 892, 829, 893, 830, 894, 831, 895, 896, 960, 897, 961, 898, 962, 899, 963, 900, 964, 901, 965, 902, 966, 903, 967, 904, 968, 905, 969, 906, 970, 907, 971, 908, 972, 909, 973, 910, 974, 911, 975, 912, 976, 913, 977, 914, 978, 915, 979, 916, 980, 917, 981, 918, 982, 919, 983, 920, 984, 921, 985, 922, 986, 923, 987, 924, 988, 925, 989, 926, 990, 927, 991, 928, 992, 929, 993, 930, 994, 931, 995, 932, 996, 933, 997, 934, 998, 935, 999, 936, 1000, 937, 1001, 938, 1002, 939, 1003, 940, 1004, 941, 1005, 942, 1006, 943, 1007, 944, 1008, 945, 1009, 946, 1010, 947, 1011, 948, 1012, 949, 1013, 950, 1014, 951, 1015, 952, 1016, 953, 1017, 954, 1018, 955, 1019, 956, 1020, 957, 1021, 958, 1022, 959, 1023, 1024, 1088, 1025, 1089, 1026, 1090, 1027, 1091, 1028, 1092, 1029, 1093, 1030, 1094, 1031, 1095, 1032, 1096, 1033, 1097, 1034, 1098, 1035, 1099, 1036, 1100, 1037, 1101, 1038, 1102, 1039, 1103, 1040, 1104, 1041, 1105, 1042, 1106, 1043, 1107, 1044, 1108, 1045, 1109, 1046, 1110, 1047, 1111, 1048, 1112, 1049, 1113, 1050, 1114, 1051, 1115, 1052, 1116, 1053, 1117, 1054, 1118, 1055, 1119, 1056, 1120, 1057, 1121, 1058, 1122, 1059, 1123, 1060, 1124, 1061, 1125, 1062, 1126, 1063, 1127, 1064, 1128, 1065, 1129, 1066, 1130, 1067, 1131, 1068, 1132, 1069, 1133, 1070, 1134, 1071, 1135, 1072, 1136, 1073, 1137, 1074, 1138, 1075, 1139, 1076, 1140, 1077, 1141, 1078, 1142, 1079, 1143, 1080, 1144, 1081, 1145, 1082, 1146, 1083, 1147, 1084, 1148, 1085, 1149, 1086, 1150, 1087, 1151, 1152, 1216, 1153, 1217, 1154, 1218, 1155, 1219, 1156, 1220, 1157, 1221, 1158, 1222, 1159, 1223, 1160, 1224, 1161, 1225, 1162, 1226, 1163, 1227, 1164, 1228, 1165, 1229, 1166, 1230, 1167, 1231, 1168, 1232, 1169, 1233, 1170, 1234, 1171, 1235, 1172, 1236, 1173, 1237, 1174, 1238, 1175, 1239, 1176, 1240, 1177, 1241, 1178, 1242, 1179, 1243, 1180, 1244, 1181, 1245, 1182, 1246, 1183, 1247, 1184, 1248, 1185, 1249, 1186, 1250, 1187, 1251, 1188, 1252, 1189, 1253, 1190, 1254, 1191, 1255, 1192, 1256, 1193, 1257, 1194, 1258, 1195, 1259, 1196, 1260, 1197, 1261, 1198, 1262, 1199, 1263, 1200, 1264, 1201, 1265, 1202, 1266, 1203, 1267, 1204, 1268, 1205, 1269, 1206, 1270, 1207, 1271, 1208, 1272, 1209, 1273, 1210, 1274, 1211, 1275, 1212, 1276, 1213, 1277, 1214, 1278, 1215, 1279]
INTERLEAVE_MAP_KV = [0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95, 32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127, 128, 192, 129, 193, 130, 194, 131, 195, 132, 196, 133, 197, 134, 198, 135, 199, 136, 200, 137, 201, 138, 202, 139, 203, 140, 204, 141, 205, 142, 206, 143, 207, 144, 208, 145, 209, 146, 210, 147, 211, 148, 212, 149, 213, 150, 214, 151, 215, 152, 216, 153, 217, 154, 218, 155, 219, 156, 220, 157, 221, 158, 222, 159, 223, 160, 224, 161, 225, 162, 226, 163, 227, 164, 228, 165, 229, 166, 230, 167, 231, 168, 232, 169, 233, 170, 234, 171, 235, 172, 236, 173, 237, 174, 238, 175, 239, 176, 240, 177, 241, 178, 242, 179, 243, 180, 244, 181, 245, 182, 246, 183, 247, 184, 248, 185, 249, 186, 250, 187, 251, 188, 252, 189, 253, 190, 254, 191, 255]

INV_INTERLEAVE_MAP_Q = inverse_perm(INTERLEAVE_MAP_Q)
INV_INTERLEAVE_MAP_KV = inverse_perm(INTERLEAVE_MAP_KV)

ACTIVATION_ALIGNMENT_EXAMPLE = """
Example of alignling activations of feed-forward layer between HF and vLLM:

We are to implement the following function to align the activations:

def postprocess_hf_activations() -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # input format: list of dicts{{"input": tensor, "kwargs": dict}}
    # output format: list of tuples
    hf_inputs = torch.load("/path/to/activations/hf_logits_output_llama4_rank0_model_layers_24_feed_forward_input.pt", weights_only=False)
    hf_outputs = torch.load("/path/to/activations/hf_logits_output_llama4_rank0_model_layers_24_feed_forward_output.pt", weights_only=False)
    vllm_inputs = torch.load("/path/to/activations/vllm_logits_output_llama4_rank0_language_model_model_layers_24_feed_forward_input.pt", weights_only=False)
    vllm_outputs = torch.load("/path/to/activations/vllm_logits_output_llama4_rank0_language_model_model_layers_24_feed_forward_output.pt", weights_only=False)
    
    # TODO: implement the function
    
    return hf_inputs, hf_outputs, vllm_inputs, vllm_outputs


We are given the following information:

1. activation previews
hf_inputs[0]['input'][0] preview Shape: (1, 6, 5120), dtype: torch.bfloat16, Tensor Preview: tensor([[[-0.0371,  0.0047, -0.0078,  0.0175,  0.0085],
         [-0.2695,  0.0630, -0.3203, -0.0148, -0.0879],
         [-0.0811, -0.1035, -0.2891, -0.6797, -0.1758],
         [ 0.5117, -0.0859, -0.1787, -0.2168, -0.4648],
         [ 0.1079, -0.1279, -0.2500, -0.3887,  0.1729]]], dtype=torch.bfloat16)
hf_outputs[0][0] preview Shape: (6, 5120), dtype: torch.bfloat16, Tensor Preview: tensor([[ 0.0225,  0.0820, -0.0020,  0.0057, -0.1191],
        [ 0.0820,  0.0801,  0.3242,  0.0282, -0.1338],
        [ 0.0125,  0.1260,  0.0386, -0.1235, -0.0527],
        [-0.1309,  0.0830,  0.1001, -0.1748, -0.1182],
        [-0.0132, -0.1475,  0.1660, -0.2490, -0.2695]], dtype=torch.bfloat16)
hf_outputs[0][1] preview Shape: (16, 6), dtype: torch.bfloat16, Tensor Preview: tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.4102, 0.4414, 0.8359],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=torch.bfloat16)
vllm_inputs[0]['input'][0] preview Shape: (6, 5120), dtype: torch.bfloat16, Tensor Preview: tensor([[-0.0371,  0.0047, -0.0078,  0.0176,  0.0085],
        [-0.2676,  0.0610, -0.3203, -0.0142, -0.0874],
        [-0.0811, -0.1006, -0.2891, -0.6719, -0.1729],
        [ 0.5078, -0.0845, -0.1729, -0.2148, -0.4609],
        [ 0.1006, -0.1289, -0.2461, -0.3809,  0.1709]], dtype=torch.bfloat16)
vllm_outputs[0][0] preview Shape: (6, 5120), dtype: torch.bfloat16, Tensor Preview: tensor([[ 0.0225,  0.0815, -0.0020,  0.0058, -0.1191],
        [ 0.0830,  0.0815,  0.3242,  0.0284, -0.1338],
        [ 0.0114,  0.1279,  0.0383, -0.1260, -0.0549],
        [-0.1289,  0.0806,  0.0996, -0.1709, -0.1162],
        [-0.0112, -0.1465,  0.1680, -0.2471, -0.2656]], dtype=torch.bfloat16)

2. class definitions of feed-forward layer, as well as its parent class in HF and vLLM
HuggingFace Module Code:
@use_kernel_forward_from_hub('Llama4TextMoe')
class Llama4TextMoe(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float('-inf')).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)
        routed_in = hidden_states.repeat(self.num_experts, 1)
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out.add_(routed_out.reshape(self.num_experts, -1, self.hidden_dim).sum(dim=0))
        return (out, router_scores)

class Llama4TextDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)
        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, output_router_logits: Optional[bool]=False, use_cache: Optional[bool]=False, cache_position: Optional[torch.LongTensor]=None, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None, **kwargs: Unpack[FlashAttentionKwargs]) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, **kwargs)
        hidden_states = residual + attention_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs

vLLM Module Code:
class Llama4MoE(nn.Module):

    def __init__(self, config: Llama4TextConfig, quant_config: Optional[QuantizationConfig]=None, prefix: str=''):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(config.hidden_size, config.num_local_experts, bias=False, quant_config=None, prefix=f'{prefix}.router')
        self.experts = FusedMoE(num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, hidden_size=config.hidden_size, custom_routing_function=Llama4MoE.custom_routing_function, intermediate_size=intermediate_size_moe, apply_router_weight_on_input=True, reduce_results=False, renormalize=False, quant_config=quant_config, prefix=f'{prefix}.experts')
        self.shared_expert = LlamaMLP(hidden_size=config.hidden_size, intermediate_size=intermediate_size_moe, hidden_act='silu', quant_config=quant_config, bias=False, prefix=f'{prefix}.shared_expert', reduce_results=False)

    def forward(self, hidden_states):
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        experts_out = routed_out + shared_out
        if self.tp_size > 1:
            experts_out = tensor_model_parallel_all_reduce(experts_out)
        return experts_out

class Llama4DecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: Llama4TextConfig, cache_config: Optional[CacheConfig]=None, quant_config: Optional[QuantizationConfig]=None, prefix: str='') -> None:
        self.layer_idx = extract_layer_index(prefix)
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.self_attn = Llama4Attention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config, bias=False, bias_o_proj=False, cache_config=cache_config, prefix=f'{prefix}.self_attn')
        is_moe_layer = (self.layer_idx + 1) % config.interleave_moe_layer_step == 0
        if is_moe_layer:
            self.feed_forward = Llama4MoE(config=config, quant_config=quant_config, prefix=f'{prefix}.feed_forward')
        else:
            self.feed_forward = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size_mlp, hidden_act='silu', quant_config=quant_config, bias=False, prefix=f'{prefix}.feed_forward')
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return (hidden_states, residual)

3. observations:
- HF's feed-forward layer's forward function def forward(self, hidden_states): returns a tuple (out, router_scores), while vLLM's feed-forward layer's forward function def forward(self, hidden_states): returns only experts_out.
- In HF, the decoder layer calls the feed-forward layer like this: hidden_states = self.feed_forward(hidden_states), and then unpacks the tuple if it's an MoE layer: if self.is_moe_layer: hidden_states, router_logits = hidden_states. In vLLM, the decoder layer calls the feed-forward layer like this: hidden_states = self.feed_forward(hidden_states), without any unpacking. Also, hidden_states is invoked as input instead of using keyword argument.
- hf_inputs[0]['input'][0] and vllm_inputs[0]['input'][0] are hidden_states input, and they are very close but not exactly the same (slight numerical differences). Also, the shape is (6, 5120) in vLLM, while in HF it's (1, 6, 5120) - an extra leading dimension. (1, 6, 5120) means (batch_size=1, seq_len=6, hidden_size=5120). We should remove this extra leading dimension in HF input to match the shape of vLLM input.
- hf_outputs[0][0] is out and hf_outputs[0][1] is router_scores from HF feed-forward layer. vllm_outputs[0][0] is experts_out from vLLM feed-forward layer. vLLM does not output router_scores. hf_outputs[0][0] and vllm_outputs[0][0] are very close but not exactly the same (slight numerical differences). They have the same shape (6, 5120), which means (seq_len=6, hidden_size=5120). They are likely the output activations.

Final code:

def postprocess_hf_activations() -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # input format: list of dicts{{"input": tensor, "kwargs": dict}}
    # output format: list of tuples
    hf_inputs = ...
    hf_outputs = ...
    vllm_inputs = ...
    vllm_outputs = ...
    
    # TODO: implement the function
    # Remove the extra leading dimension in hf_inputs
    hf_inputs = [hf_inputs[i]["input"][0].squeeze(0) for i in range(len(hf_inputs))]
    # Extract the out from hf_outputs, which is the first element of the tuple
    hf_outputs = [hf_outputs[i][0] for i in range(len(hf_outputs))]
    # Extract the input tensor from vllm_inputs
    vllm_inputs = [vllm_inputs[i]["input"][0] for i in range(len(vllm_inputs))]
    # Extract the experts_out from vllm_outputs, which is the first element of the tuple
    vllm_outputs = [vllm_outputs[i][0] for i in range(len(vllm_outputs))]
    
    return hf_inputs, hf_outputs, vllm_inputs, vllm_outputs

"""