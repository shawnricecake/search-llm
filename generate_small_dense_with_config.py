import torch
from lib_utils.utils import skip, decode_cand_tuple
from model.module.linear_super import sample_weight

sample_config = [32, 10496, 10496, 10496, 9984, 11008, 8960, 10496, 6400, 5376, 7936, 7936, 5888, 9984, 6912, 4864, 4864, 8960, 6400, 5376, 7936, 7936, 6400, 6912, 4864, 6912, 5376, 5376, 6400, 8448, 5888, 10496, 8960, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 0.13716425089140216, 0.9989146101501304, 0.20495626014238122, 0.4376342622440923, 0.3774073300110513, 0.5557324124364033, 0.3871143289503648, 0.7709398590094507, 0.5961088920453859, 0.8434812283726285, 0.5145565293780437, 0.08846661741053996, 0.8848786671044369, 0.9078977631417641, 0.46213415470978714, 0.7215095795951649, 0.8504222462544468, 0.4302025386540399, 0.1234390493112244, 0.2998590622322622, 0.8331215268661771, 0.6534756462933968, 0.5744785622411066, 0.7416904944462073, 0.2801099934812795, 0.5449772205522283, 0.9976629909443853, 0.9161786370109023, 0.7029700158352867, 0.0799920493964198, 0.5665912711485945, 0.9909763354048148]
output_small_dense_ckpt_path = ""

model_path = "llama-1-7b"
compensated_checkpoints = ""

sample_layer_num, sample_intermediate_size, sample_num_attention_heads, \
        sample_attention_head_dim, sample_weight_start = decode_cand_tuple(sample_config)
sample_hidden_size = []
for i in range(sample_layer_num):
    sample_hidden_size.append(int(sample_attention_head_dim[i] * sample_num_attention_heads[i]))

torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

# create original model
from model.llama_supernet import LlamaForCausalLM
model_original = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model_original.seqlen = 2048
model_original.model.config.use_cache = False
if compensated_checkpoints != "":
    model_original.load_state_dict(torch.load(compensated_checkpoints))

# create small dense
from model.llama_small_dense import LlamaForCausalLM as LlamaForCausalLM_small
model_small = LlamaForCausalLM_small(model_original.model.config, sample_config)
model_small.seqlen = 2048
model_small.model.config.use_cache = False


# start merging the weights into the small dense
original_state_dict = model_original.state_dict()
small_state_dict = model_small.state_dict()
layer_idx = 0
print("start merging...")
for name in original_state_dict:
    print("merging {}..".format(name))
    if 'embed_tokens' in name:
        small_state_dict[name] = original_state_dict[name]
    elif 'norm' in name:
        small_state_dict[name] = original_state_dict[name]
    elif 'lm_head' in name:
        small_state_dict[name] = original_state_dict[name]
    elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
        # todo: implement for head sparsity
        small_state_dict[name] = original_state_dict[name]
    elif 'layernorm' in name:
        small_state_dict[name] = original_state_dict[name]
    elif 'gate_proj' in name or 'up_proj' in name:
        sampled_weight = sample_weight(
            original_state_dict[name],
            sample_hidden_size[layer_idx],
            sample_intermediate_size[layer_idx],
            sample_weight_start[layer_idx]
        )
        small_state_dict[name] = sampled_weight
    elif 'down_proj' in name:
        sampled_weight = sample_weight(
            original_state_dict[name],
            sample_intermediate_size[layer_idx],
            sample_hidden_size[layer_idx],
            sample_weight_start[layer_idx]
        )
        small_state_dict[name] = sampled_weight
        pass
    else:
        raise ValueError("did not implement for {}".format(name))

    if 'post_attention_layernorm' in name:
        layer_idx += 1

print("weight merging done!")
torch.save(small_state_dict, output_small_dense_ckpt_path)


