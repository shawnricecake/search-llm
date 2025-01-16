# Search for Efficient LLMs

Official repo for paper: [Search for Efficient Large Language Models](https://arxiv.org/pdf/2409.17372)

This paper is accepted by NeurIPS 2024

## Usage
0. Set initialization (TODO)
1. Revise search space at `experiments/llama-7b.yaml`
2. Directly start search with `sh run-search-llama.sh`

## Citation
```
@inproceedings{
    shen2024search,
    title     = {Search for Efficient Large Language Models},
    author    = {Shen, Xuan and Zhao, Pu and Gong, Yifan and Kong, Zhenglun and Zhan, Zheng and Wu, Yushu and Lin, Ming and Wu, Chao and Lin, Xue and Wang, Yanzhi},
    booktitle = {NeurIPS},
    year      = {2024},
}
```

## Acknowledgment
The code is mainly based on the NAS work [AutoFormer](https://github.com/microsoft/Cream).
