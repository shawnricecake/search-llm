import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import argparse
import os
import yaml

from lib_utils import utils
from lib_utils.utils import skip
from lib_utils.config import cfg, update_config_from_file
from lib_utils.data_utils import get_loaders
from lib_utils.evaluate import evaluate
from lib_utils.multi_gpu_utils import llama_multigpu, opt_multigpu
from lib_utils.utils import decode_cand_tuple


class EvolutionSearcher(object):
    def __init__(
            self, args, device, model, model_without_ddp,
            choices, mutation_choices,
            val_loader, test_loader,
            output_dir
    ):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices
        self.mutation_choices = mutation_choices

    def save_checkpoint(self):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

        return info

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):  # to revise hidden size
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        sample_layer_num, sample_intermediate_size, sample_num_attention_heads, \
        sample_attention_head_dim, sample_weight_start = decode_cand_tuple(cand)

        sampled_config = {}
        sampled_config['sample_layer_num'] = sample_layer_num
        sampled_config['sample_intermediate_size'] = sample_intermediate_size
        sampled_config['sample_num_attention_heads'] = sample_num_attention_heads
        sampled_config['sample_attention_head_dim'] = sample_attention_head_dim
        sample_hidden_size = []
        for i in range(sample_layer_num):
            sample_hidden_size.append(int(sample_attention_head_dim[i] * sample_num_attention_heads[i]))
        sampled_config['sample_hidden_size'] = sample_hidden_size
        sampled_config['sample_weight_start'] = sample_weight_start
        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info['params'] = n_parameters / 10. ** 6

        if info['params'] > self.parameters_limits:
            # print('parameters limit exceed: {:.2f}'.format(info['params']))
            return False

        if info['params'] < self.min_parameters_limits:
            # print('under minimum parameters limit : {:.2f}'.format(info['params']))
            return False

        # use training dataset here
        eval_ppl = evaluate(self.model, self.val_loader, self.device, full_eval=False)
        # test_ppl = evaluate(self.model, self.test_loader, self.device, full_eval=False)
        test_ppl = eval_ppl

        # logging
        print(
            "logging-: "
            "params: {:.2f} M; "
            "ppl: {:.2f}; "
            "num layer: {}; "
            "hidden_size: {}; "
            "intermediate_size: {}; "
            "attention_heads: {}; "
            "head dim: {}; "
            "start location: {};".format(
                info['params'],
                eval_ppl,
                sample_layer_num,
                sample_hidden_size,
                sample_intermediate_size,
                sample_num_attention_heads,
                sample_attention_head_dim,
                sample_weight_start
            )
        )

        info['acc'] = eval_ppl * (-1)
        info['test_acc'] = test_ppl * (-1)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ['sample_intermediate_size', 'sample_num_attention_heads', 'sample_attention_head_dim']
        depth = random.choice(self.choices['sample_layer_num'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))
        for i in range(depth):
            cand_tuple.append(random.uniform(0, 1))
        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        iter = 0
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            sample_layer_num, sample_intermediate_size, sample_num_attention_heads, \
            sample_attention_head_dim, sample_weight_start = decode_cand_tuple(cand)

            # new_sample_layer_num
            random_s = random.random()
            if random_s < s_prob:
                new_sample_layer_num = random.choice(self.mutation_choices['sample_layer_num'])

                if new_sample_layer_num > sample_layer_num:
                    # large mutation
                    sample_intermediate_size = sample_intermediate_size + [random.choice(
                        self.mutation_choices['sample_intermediate_size']
                    ) for _ in range(new_sample_layer_num - sample_layer_num)]
                    sample_num_attention_heads = sample_num_attention_heads + [random.choice(
                        self.mutation_choices['sample_num_attention_heads']
                    ) for _ in range(new_sample_layer_num - sample_layer_num)]
                    sample_attention_head_dim = sample_attention_head_dim + [random.choice(
                        self.mutation_choices['sample_attention_head_dim']
                    ) for _ in range(new_sample_layer_num - sample_layer_num)]
                    sample_weight_start = sample_weight_start + [
                        random.uniform(0, 1)
                        for _ in range(new_sample_layer_num - sample_layer_num)]

                    # small mutation
                    # sample_intermediate_size = sample_intermediate_size + [
                    #     self.mutation_choices['sample_intermediate_size'][-1]
                    #     for _ in range(new_sample_layer_num - sample_layer_num)]
                    # sample_num_attention_heads = sample_num_attention_heads + [
                    #     self.mutation_choices['sample_num_attention_heads'][-1]
                    #     for _ in range(new_sample_layer_num - sample_layer_num)]
                    # sample_attention_head_dim = sample_attention_head_dim + [
                    #     self.mutation_choices['sample_attention_head_dim'][-1]
                    #     for _ in range(new_sample_layer_num - sample_layer_num)]
                    # sample_weight_start = sample_weight_start + [
                    #     random.uniform(0,1) for _ in range(new_sample_layer_num - sample_layer_num)]
                else:
                    sample_intermediate_size = sample_intermediate_size[:new_sample_layer_num]
                    sample_num_attention_heads = sample_num_attention_heads[:new_sample_layer_num]
                    sample_attention_head_dim = sample_attention_head_dim[:new_sample_layer_num]
                    sample_weight_start = sample_weight_start[:new_sample_layer_num]

                sample_layer_num = new_sample_layer_num

            # large mutation
            # sample_intermediate_size
            for i in range(sample_layer_num):
                random_s = random.random()
                if random_s < m_prob:
                    sample_intermediate_size[i] = random.choice(self.mutation_choices['sample_intermediate_size'])
            # sample_num_attention_heads
            for i in range(sample_layer_num):
                random_s = random.random()
                if random_s < m_prob:
                    sample_num_attention_heads[i] = random.choice(self.mutation_choices['sample_num_attention_heads'])
            # sample_attention_head_dim
            for i in range(sample_layer_num):
                random_s = random.random()
                if random_s < m_prob:
                    sample_attention_head_dim[i] = random.choice(self.mutation_choices['sample_attention_head_dim'])
            # sample_weight_start
            for i in range(sample_layer_num):
                random_s = random.random()
                if random_s < m_prob:
                    sample_weight_start[i] = random.uniform(0, 1)

            # small mutation
            # random1_type = random.choice(
            #     ["sample_intermediate_size",
            #      "sample_num_attention_heads",
            #      "sample_attention_head_dim",
            #      "sample_weight_start"]
            # )
            # random2_layer = random.choice([x for x in range(sample_layer_num)])
            # random3_value = random.choice(self.mutation_choices[random1_type]) \
            #     if random1_type != "sample_weight_start" else random.uniform(0, 1)
            # if random1_type == "sample_intermediate_size":
            #     current_value = sample_intermediate_size[random2_layer]
            # elif random1_type == 'sample_num_attention_heads':
            #     current_value = sample_num_attention_heads[random2_layer]
            # elif random1_type == "sample_attention_head_dim":
            #     current_value = sample_attention_head_dim[random2_layer]
            # elif random1_type == "sample_weight_start":
            #     current_value = sample_weight_start[random2_layer]
            # else:
            #     raise ValueError
            # while current_value == random3_value:
            #     random3_value = random.choice(self.mutation_choices[random1_type]) \
            #     if random1_type != "sample_weight_start" else random.uniform(0, 1)
            # if random1_type == "sample_intermediate_size":
            #     sample_intermediate_size[random2_layer] = random3_value
            # elif random1_type == 'sample_num_attention_heads':
            #     sample_num_attention_heads[random2_layer] = random3_value
            # elif random1_type == "sample_attention_head_dim":
            #     sample_attention_head_dim[random2_layer] = random3_value
            # else:
            #     raise ValueError

            result_cand = [sample_layer_num] + sample_intermediate_size + sample_num_attention_heads \
                          + sample_attention_head_dim + sample_weight_start

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'
            .format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        total_info = []

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.get_random(self.population_num)

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])

            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[self.select_num])))

            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[self.select_num]):

                sample_layer_num, sample_intermediate_size, sample_num_attention_heads, \
                sample_attention_head_dim, sample_weight_start = decode_cand_tuple(cand)

                if hasattr(self.model.model, "num_attention_heads"):   # llama
                    if sum(sample_num_attention_heads) < self.model.model.num_attention_heads * sample_layer_num or \
                            sum(sample_attention_head_dim) < \
                            (self.model.model.hidden_size // self.model.model.num_attention_heads) * sample_layer_num:
                        head_sparsity = True
                    else:
                        head_sparsity = False
                elif hasattr(self.model.model, "decoder"):  # opt
                    if sum(sample_num_attention_heads) < self.model.config.num_attention_heads * sample_layer_num or \
                            sum(sample_attention_head_dim) < \
                            (self.model.config.hidden_size // self.model.config.num_attention_heads) * sample_layer_num:
                        head_sparsity = True
                    else:
                        head_sparsity = False
                else:
                    head_sparsity = "NotSupport"

                # print('No.{}: val ppl={:.2f}, test ppl={:.2f}, params={:.2f}, {}'.format(
                print('No.{}: val ppl={:.2f}, params={:.2f}, head_sparsity={}, {}'.format(
                    i + 1,
                    self.vis_dict[cand]['acc'] * (-1),
                    # self.vis_dict[cand]['test_acc'] * (-1),
                    self.vis_dict[cand]['params'],
                    head_sparsity,
                    cand
                ))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])

            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            self.candidates = mutation + crossover

            self.epoch += 1

            info = self.save_checkpoint()
            total_info.append(info)


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=18)

    # config file
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    # Normal
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs.')
    parser.add_argument('--model-path', type=str, default=None, help='model path')
    parser.add_argument('--dataset', type=str, default=None, help='dataset')

    return parser


def main(args):
    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        f.write(args_text)
    # fix the seed for reproducibility

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    # load model ===============================================================
    model_path = args.model_path
    print("Loading Pretrained LLMs from {}".format(model_path))
    print(cfg)
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
    print("Original model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    # load model ===============================================================
    # model.to(device)
    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []
    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]

    if 'llama' in model_path:
        llama_multigpu(model, gpus, gpu_dist)
    elif 'opt' in model_path:
        opt_multigpu(model, gpus, gpu_dist)

    model_without_ddp = model

    # dataset ===============================================================
    dataset_name = args.dataset
    _, data_loader_train, data_loader_test = get_loaders(
        dataset_name, seed=args.seed, model=model_path, seqlen=model.seqlen
    )
    # dataset ===============================================================

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("resume from checkpoint: {}".format(args.resume))
        model_without_ddp.load_state_dict(checkpoint['model'])

    sample_attention_head_dim_here = [x for x in range(
        cfg.random_space['sample_attention_head_dim'][0],
        cfg.random_space['sample_attention_head_dim'][1] + cfg.random_space['sample_attention_head_dim'][2],
        cfg.random_space['sample_attention_head_dim'][2]
    )]
    sample_num_attention_heads_here = [x for x in range(
        cfg.random_space['sample_num_attention_heads'][0],
        cfg.random_space['sample_num_attention_heads'][1] + cfg.random_space['sample_num_attention_heads'][2],
        cfg.random_space['sample_num_attention_heads'][2]
    )]
    sample_intermediate_size_here = [x for x in range(
        cfg.random_space['sample_intermediate_size'][0],
        cfg.random_space['sample_intermediate_size'][1] + cfg.random_space['sample_intermediate_size'][2],
        cfg.random_space['sample_intermediate_size'][2]
    )]
    sample_layer_num_here = [x for x in range(
        cfg.random_space['sample_layer_num'][0],
        cfg.random_space['sample_layer_num'][1] + cfg.random_space['sample_layer_num'][2],
        cfg.random_space['sample_layer_num'][2]
    )]
    choices = {
        "sample_attention_head_dim": sample_attention_head_dim_here,
        "sample_num_attention_heads": sample_num_attention_heads_here,
        "sample_intermediate_size": sample_intermediate_size_here,
        "sample_layer_num": sample_layer_num_here,
    }
    sample_attention_head_dim_here = [x for x in range(
        cfg.mutation_space['sample_attention_head_dim'][0],
        cfg.mutation_space['sample_attention_head_dim'][1] + cfg.mutation_space['sample_attention_head_dim'][2],
        cfg.mutation_space['sample_attention_head_dim'][2]
    )]
    sample_num_attention_heads_here = [x for x in range(
        cfg.mutation_space['sample_num_attention_heads'][0],
        cfg.mutation_space['sample_num_attention_heads'][1] + cfg.mutation_space['sample_num_attention_heads'][2],
        cfg.mutation_space['sample_num_attention_heads'][2]
    )]
    sample_intermediate_size_here = [x for x in range(
        cfg.mutation_space['sample_intermediate_size'][0],
        cfg.mutation_space['sample_intermediate_size'][1] + cfg.mutation_space['sample_intermediate_size'][2],
        cfg.mutation_space['sample_intermediate_size'][2]
    )]
    sample_layer_num_here = [x for x in range(
        cfg.mutation_space['sample_layer_num'][0],
        cfg.mutation_space['sample_layer_num'][1] + cfg.mutation_space['sample_layer_num'][2],
        cfg.mutation_space['sample_layer_num'][2]
    )]
    mutation_choices = {
        "sample_attention_head_dim": sample_attention_head_dim_here,
        "sample_num_attention_heads": sample_num_attention_heads_here,
        "sample_intermediate_size": sample_intermediate_size_here,
        "sample_layer_num": sample_layer_num_here,
    }

    t = time.time()
    searcher = EvolutionSearcher(
        args, device, model, model_without_ddp,
        choices, mutation_choices,
        data_loader_train,
        data_loader_test,
        args.output_dir
    )

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLMs Subnet Evolution Search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
