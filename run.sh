export CUDA_VISIBLE_DEVICES=5
python3 ./pred_hf.py --model Llama-3.1-8B-Instruct --generation_prompt --n_proc 1
python3 ./result.py