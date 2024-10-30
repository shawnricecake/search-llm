import torch
from lib_utils.multi_gpu_utils import llama_multigpu, opt_multigpu
from lib_utils.data_utils import get_loaders
from lib_utils.evaluate import evaluate
from lib_utils.compensate_evaluate import llama_sequential, llama_eval
from evolution import decode_cand_tuple
from lib_utils.utils import skip, get_sample_config


# load model ===============================================================
model_path = ""        # todo
nsamples = 128
output_model_weights_path = "outputs/compensated_model.pth"
print("Loading Pretrained LLMs from {}".format(model_path))
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip
if 'llama' in model_path:
    from model.llama_supernet import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.seqlen = 2048
    model.model.config.use_cache = False
else:
    raise ValueError("Did not support this kind of model or you did not enroll the llama or opt in model path.")
# load model ===============================================================
model_without_ddp = model

DEV = torch.device('cuda:{}'.format(0))

# dataset ===============================================================
dataset_name = "wikitext2"      # todo: wikitext2, ptb, c4, bookcorpus(seqlen=128)
data_loader_train, _, data_loader_test = get_loaders(
    dataset_name,
    nsamples=nsamples,
    seed=123, model=model_path, seqlen=model.seqlen
)
# dataset ===============================================================

# compensate weights ======================================================
if 'llama' in model_path:
    llama_sequential(model, data_loader_train, DEV, nsamples=nsamples)
# compensate weights ======================================================

# save model weights ======================================================
torch.save(model.state_dict(), output_model_weights_path)
print("Compensation finished.")
# save model weights ======================================================



