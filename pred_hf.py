import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch.multiprocessing as mp
import torch

# 加载模型路径和最大长度映射
model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())
dataset_path = json.loads(open("config/dataset2path.json", encoding='utf-8').read())

# 读取模板
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

@torch.inference_mode()
def generate_response(model, input_ids, max_new_tokens, temperature, stop, chunk_size=32768):
    assert temperature == 0.0
    assert len(input_ids) > 0

    device = model.device

    prefill_len = len(input_ids) - 1
    kv_cache = DynamicCache()

    for chunk_begin in range(0, prefill_len, chunk_size):
        chunk_end = min(chunk_begin + chunk_size, prefill_len)
        kv_cache = model(torch.tensor(input_ids[chunk_begin:chunk_end], dtype=torch.long).unsqueeze(0).to(device), past_key_values=kv_cache).past_key_values

    output_ids = [input_ids[-1]]

    for _ in range(max_new_tokens):
        output = model(torch.tensor(output_ids[-1:]).unsqueeze(0).to(device), past_key_values=kv_cache)
        kv_cache = output.past_key_values
        output_ids.append(output.logits[0].argmax())
        if output_ids[-1] == stop:
            break

    return output_ids[1:]

def query_llm(prompt, model, tokenizer, max_len, device, temperature=0.0, max_new_tokens=128, stop=None):
    """
    使用 HuggingFace 本地模型生成响应。
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids =input_ids[:max_len//2] + input_ids[-max_len//2:]

    # 生成输出
    output_ids = generate_response(
        model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=tokenizer.eos_token_id
    )
    # 解码生成的 tokens
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

def get_pred(data, args, save_path, rank):
    """
    每个进程加载自己的模型和分词器，然后处理数据。
    """
    model_name = args.model
    model_path = model_map.get(model_name, model_name)
    max_len = maxlen_map.get(model_name)  # 默认最大长度

    # 根据模型名称选择分词器
    if "gpt" in model_name.lower() or "o1" in model_name.lower():
        # 假设这些模型使用同一种分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    device = f"cuda:{rank}"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()

    with open(save_path, 'a', encoding='utf-8') as fout:
        for item in tqdm(data, desc=f"Process {rank}"):
            context = item['context']
            if args.rag > 0:
                template = template_rag
                retrieved = item.get("retrieved_context", [])[:args.rag]
                retrieved = sorted(retrieved, key=lambda x: x.get('c_idx', 0))
                context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
            elif args.no_context:
                template = template_no_context
            elif args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot

            prompt = template.replace('$DOC$', context.strip()) \
                            .replace('$Q$', item['question'].strip()) \
                            .replace('$C_A$', item['choice_A'].strip()) \
                            .replace('$C_B$', item['choice_B'].strip()) \
                            .replace('$C_C$', item['choice_C'].strip()) \
                            .replace('$C_D$', item['choice_D'].strip())
            
            generation_prompt = r"The correct answer is ("

            if args.cot:
                output = query_llm(prompt, model, tokenizer, device, temperature=0.0, max_new_tokens=1024)
            else:
                if args.generation_prompt:
                    output = query_llm(prompt + generation_prompt, model, tokenizer, max_len, device, temperature=0.0, max_new_tokens=1)
                else:
                    output = query_llm(prompt, model, tokenizer, max_len, device, temperature=0.0, max_new_tokens=128)

            # if output == '':
            #     continue

            if args.cot:
                # 提取链式思考的回答
                response = output.strip()
                item['response_cot'] = response
                prompt_ans = template_0shot_cot_ans.replace('$DOC$', context.strip()) \
                                                   .replace('$Q$', item['question'].strip()) \
                                                   .replace('$C_A$', item['choice_A'].strip()) \
                                                   .replace('$C_B$', item['choice_B'].strip()) \
                                                   .replace('$C_C$', item['choice_C'].strip()) \
                                                   .replace('$C_D$', item['choice_D'].strip()) \
                                                   .replace('$COT$', response)
                if args.generation_prompt: 
                    output = query_llm(prompt_ans + generation_prompt, model, tokenizer, device, temperature=0.0, max_new_tokens=1)
                else:
                    output = query_llm(prompt_ans, model, tokenizer, device, temperature=0.0, max_new_tokens=128)

                # if output == '':
                #     continue

            response = output.strip()
            item['response'] = response
            item['pred'] = response if args.generation_prompt else extract_answer(response)
            item['judge'] = item['pred'] == item.get('answer', '').strip()
            item['context'] = context[:1000]  # 保留部分上下文

            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--generation_prompt", action="store_true", default=False)
    parser.add_argument("--cot", "-cot", action='store_true')  # 使用链式思考
    parser.add_argument("--no_context", "-nc", action='store_true')  # 不使用上下文
    parser.add_argument("--rag", "-rag", type=int, default=0)  # 使用 RAG 时设置
    parser.add_argument("--n_proc", "-n", type=int, default=16)  # 进程数
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[INFO] Runing with args: {args}")

    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    # 加载数据集
    dataset = load_dataset(dataset_path["longbench-v2"], split='train')
    data_all = [{
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item.get("context", "")
    } for item in dataset]

    # 读取已处理的数据，避免重复
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    # 过滤需要处理的数据
    data = [item for item in data_all if item["_id"] not in has_data]
    print(f"[INFO] Total data to process: {len(data)}")

    # 分割数据到各个进程
    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]

    # 启动多进程
    # processes = []
    # for rank in range(args.n_proc):
    #     p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, rank))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    get_pred(data_subsets[0], args, out_file, 0)

    print("[INFO] All processes completed.")

if __name__ == "__main__":
    main()
