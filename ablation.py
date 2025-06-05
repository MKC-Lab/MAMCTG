import argparse
from main import generate
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import json
import torch


print("loading data...")



with open('autodl-fs/processed_senticap.json', 'r') as f:
    data = json.load(f)['test']



# --lm_path autodl-tmp/gemma-7b-it \
# --clip_path autodl-tmp/clip-vit-base-patch32 \
# --output_path . \
# --caption_dataset_path autodl-fs/processed_senticap.json \
# --image_dataset_folder_path autodl-tmp/senticap_dataset/senticap_images \

# polish_num = 5
# desired_attr = "Humorous"
device = "cuda" if torch.cuda.is_available() else "cpu"
# desired_attr = "Humorous"


# 初始化模型和tokenizer
print("loading language model...")
tokenizer = AutoTokenizer.from_pretrained('autodl-tmp/gemma-7b-it')
model = AutoModelForCausalLM.from_pretrained('autodl-tmp/gemma-7b-it')
model.to(device)

# 加载CLIP模型和processor
print("loading CLIP model...")
clip_model = CLIPModel.from_pretrained('autodl-tmp/clip-vit-base-patch32')
clip_model.to(device)
clip_processor = CLIPProcessor.from_pretrained('autodl-tmp/clip-vit-base-patch32')



for top_k in [200, 100, 20]:
    for nrq in [60, 30, 5]:
        for time in [10]:

            parser = argparse.ArgumentParser()
            parser.add_argument("--lm_type", default='encoder_decoder', type=str)
            parser.add_argument("--lm_path", default='autodl-tmp/gemma-7b-ite', type=str)
            parser.add_argument("--clip_path", default='autodl-tmp/clip-vit-base-patch32', type=str)
            parser.add_argument("--output_path", default='.', type=str)
            parser.add_argument("--caption_dataset_path", default='autodl-fs/processed_senticap.json', type=str)
            parser.add_argument("--image_dataset_folder_path", default='autodl-tmp/senticap_dataset/senticap_images', type=str)

            parser.add_argument("--device", default='cuda', type=str)

            parser.add_argument("--polish_times", default=time, type=int)
            parser.add_argument("--top_k", default=top_k, type=int)
            parser.add_argument("--num_return_sequences", default=nrq, type=int)

            args = parser.parse_args()

            try:
                print(f"starting generation! top_k: {top_k}, num_return_sequences: {nrq}, polish_times: {time}")
                generate(args, model, tokenizer, clip_model, clip_processor, data, desired_attrs=['Negative', 'Positive'])
            except Exception as e:
                print(f"error: {e}")
                continue





