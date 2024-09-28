docker run --name rutube_vllm \
                --runtime nvidia \
                --gpus all \
                -p 8000:8000 \
                --ipc=host \
                vllm/vllm-openai:latest \
                --model Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24 \
                --gpu-memory-utilization 0.7 \
                --tensor_parallel_size 2 \
                --max_model_len 3000