
model_path="llama-1-7b"
config="./experiments/llama-7b.yaml"

dataset="wikitext2"

python3 evolution.py \
        --model-path $model_path \
        --dataset $dataset \
        --output_dir outputs \
        --max-epochs 30 \
        --select-num 300 \
        --population-num 500 \
        --crossover-num 100 \
        --mutation-num 350 \
        --cfg $config \
        --min-param-limits 5000 \
        --param-limits 5500 \
        --seed 666


