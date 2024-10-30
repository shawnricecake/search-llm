import torch
from lib_utils.multi_gpu_utils import llama_multigpu, opt_multigpu
from lib_utils.data_utils import get_loaders
from lib_utils.evaluate import evaluate
from evolution import decode_cand_tuple
from lib_utils.utils import skip, get_sample_config


# load model ===============================================================
model_path = ""     # todo
model_compensation_weights = ""
print("Loading Pretrained LLMs from {}".format(model_path))
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip
if 'llama' in model_path:
    from model.llama_supernet import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.seqlen = 2048
    model.model.config.use_cache = False
elif 'opt' in model_path:
    from model.opt_supernet import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')  # 'auto'
    model.seqlen = model.config.max_position_embeddings
    model.model.config.use_cache = False
else:
    raise ValueError("Did not support this kind of model or you did not enroll the llama or opt in model path.")

if len(model_compensation_weights) > 0 and model_compensation_weights is not None:
    model.load_state_dict(torch.load(model_compensation_weights))
    print("Load compensated weights from {}".format(model_compensation_weights))
# load model ===============================================================

gpu_dist = [4,4,4,4,4,4,4,4]
gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
if 'llama' in model_path:
    llama_multigpu(model, gpus, gpu_dist)
elif 'opt' in model_path:
    opt_multigpu(model, gpus, gpu_dist)

model_without_ddp = model

# input config ===============================================================
config = []
# set config ===============================================================
sampled_config = get_sample_config(config)
n_parameters = model_without_ddp.get_sampled_params_numel(sampled_config)
print("model parameters: {:.2f} M".format(n_parameters / 1e6))
# set config ===============================================================

DEV = torch.device('cuda:{}'.format(0))

dataset_name = ["wikitext2", "ptb", "c4"]
for dataset in dataset_name:
    print("Dataset: {}".format(dataset))

    _, _, data_loader_test = get_loaders(dataset, seed=123, model=model_path, seqlen=model.seqlen)

    ppl = evaluate(model, data_loader_test, DEV, full_eval=True)

    print("Perplexity on dataset {}: {:.2f}".format(dataset, ppl))



