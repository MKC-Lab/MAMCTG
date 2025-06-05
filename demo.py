import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
from PIL import Image
import argparse
import json
import os
import json
import tqdm
import random


# def load_data(args, data_type, split='train'):

#     if data_type == 'flickr30k' or data_type == 'senticap':
#         with open(args.caption_dataset_path, 'r') as f:
#             data = json.load(f)['images']
  
#         splited_data = [item for item in data if item['split'] == split]
#         return splited_data
    
#     else:
#         raise NotImplementedError

def load_data(args, split='train'):
    with open(args.caption_dataset_path, 'r') as f:
        data = json.load(f)[split]

    return data



def generate(args, model, tokenizer, clip_model, clip_processor, image_path, desired_attr='positive'):



    def top_k_pairs(sentence_prob_pairs, k):
        # 按照概率值排序，然后取最后k个元素
        top_k = sorted(sentence_prob_pairs, key=lambda x: x[1])[-k:]
        # 因为我们想要的是最大的k个元素，所以需要反转列表
        top_k.reverse()
        return top_k
        
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"start generation top_k: {args.top_k}, num_return_sequences: {args.num_return_sequences}, polish_times: {args.polish_times}")

    polish_times = args.polish_times

    # 加载图像
    # image_path = "这里是你的图像路径"
    # image = Image.open("COCO_val2014_000000572811.jpg")


    desired_attr = desired_attr.lower()
        

    if desired_attr == "normal":
        desired_attr_prompt = ""
    else:
        desired_attr_prompt = f" that is " + desired_attr.lower()

    results = []
            
        
            



    if not os.path.exists(image_path):
        print("no such image.")
        return
                
    image = Image.open(image_path)

    if args.debug:
        print(image_path)


    # prompt = "Generate some keywords randomly about an image, just give me the answer"

    # bad_words = ['photo', 'image', 'photos', 'images', 'picture', 'pictures', 'photograph', 'photo of', 'image of', 'photos of', 'images of', 'picture of', 'pictures of', 'photograph of']
    bad_words = ['photo', 'image', 'photos', 'images', 'picture', 'pictures', 'photograph', 'the', 'this']
            
    # 将禁止词转换为索引
    bad_words_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]
    if args.debug:
        print(bad_words_ids)


    if args.keyword_first:

        # print("kw first")


        if args.lm_type == 'encoder_decoder':

                    
            prompt = "Generate a word about an image."
                        
        else:

            prompt = "Generate a word about an image."
                
        inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True, max_length=500).to(args.device)
                # inputs.half()
        outputs = model.generate(inputs, max_new_tokens=4, num_return_sequences=200, do_sample=True, top_k=args.top_k, bad_words_ids=bad_words_ids)
                
        inputs_len = len(inputs[0])
                
        del inputs
        torch.cuda.empty_cache()

        if args.lm_type == 'encoder_decoder':
            generated_words = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            generated_words = tokenizer.batch_decode(outputs[:, inputs_len:], skip_special_tokens=True)
                
        del outputs
        torch.cuda.empty_cache()

        # generated_words = [word.replace("image of ", "") for word in generated_words]

        clip_inputs = clip_processor(text=generated_words, images=image, return_tensors="pt", padding=True).to(args.device)
                
        # truncation
        if len(clip_inputs['input_ids'][0]) > 77:
            clip_inputs['input_ids'] = clip_inputs['input_ids'][:, :77]
            clip_inputs['attention_mask'] = clip_inputs['attention_mask'][:, :77]
                
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image
        # print(logits_per_image)
                    
        probs = logits_per_image
        # probs = logits_per_image.softmax(dim=1)
        # print(probs)
        # logits_per_image.softmax(dim=-1)

        word_prob_pairs = list(zip(generated_words, list(probs[0])))

        top_k_words = tuple(zip(*top_k_pairs(word_prob_pairs, 10)))[0]


        if args.debug:
            print(f'top_k: {top_k_words}')

        # selected_words = random.sample(top_k_words, 5)

        # print(f'top_k: {top_k_words} selected {selected_words}')


    # # save captions for the current one image with different attribute(style/sentiment)
    # item_results = []

    # previous_sentences = ["a woman takes a bread, and ready to eat.", "bread"]
    previous_sentences = []


    for polish_ind in range(polish_times):

        if args.keyword_first:

            if args.lm_type == 'encoder_decoder':

                if previous_sentences == []:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + "."
                    prompt = "Generate an image caption" + desired_attr_prompt + ", with following keywords: " + ", ".join(top_k_words) + ".\n"
                else:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \"" + "\"\n\"".join(previous_sentences) + "\" "
                    prompt = "Generate an image caption" + desired_attr_prompt + ", with following keywords: " + ", ".join(top_k_words) + "\nbased on the following multiple sentences, but modify them as you want. The following sentences are: \n\"" + "\"\n\"".join(previous_sentences) + "\"\n"
                    # prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"" + ("\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"").join(previous_sentences) + "\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer:"
                        
            else:

                raise NotImplementedError

                        # # 生成top-k的句子结果
                        # if previous_sentences == []:
                        #     # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + "."
                        #     prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\n Answer: \""
                        # else:
                        #     # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \"" + "\"\n\"".join(previous_sentences) + "\" "
                        #     # prompt = "Question: Generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \n\"" + "\"\n\"".join(previous_sentences) + "\"\n\" Answer:"
                        #     prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"" + ("\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"").join(previous_sentences) + "\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer:"
                            

            # top_k_words
            if args.debug:
                print(prompt)

        else:

            if args.lm_type == 'encoder_decoder':

                if previous_sentences == []:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + "."
                    prompt = "Generate an image caption" + desired_attr_prompt + ".\n"
                else:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \"" + "\"\n\"".join(previous_sentences) + "\" "
                    prompt = "Generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \n\"" + "\"\n\"".join(previous_sentences) + "\"\n"
                    # prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"" + ("\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"").join(previous_sentences) + "\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer:"
                        
            else:

                # 生成top-k的句子结果
                if previous_sentences == []:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + "."
                    prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\n Answer: \""
                else:
                    # prompt = "Answering directly, generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \"" + "\"\n\"".join(previous_sentences) + "\" "
                    # prompt = "Question: Generate an image caption" + desired_attr_prompt + ", based on the following multiple sentences, but modify them as you want. The following sentences are: \n\"" + "\"\n\"".join(previous_sentences) + "\"\n\" Answer:"
                    prompt = "Question: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"" + ("\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer: \"").join(previous_sentences) + "\"\nQuestion: Generate an image caption" + desired_attr_prompt + ".\nAnswer:"
                            
                    # print(prompt)
                    
            # print()
            # print(f'The input is "{prompt}"')
                
        inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True, max_length=500).to(args.device)
        # inputs.half()
        outputs = model.generate(inputs, max_new_tokens=50, num_return_sequences=args.num_return_sequences, do_sample=True, top_k=args.top_k)
                
        inputs_len = len(inputs[0])
                
        del inputs
        torch.cuda.empty_cache()

        if args.lm_type == 'encoder_decoder':
            generated_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            generated_sentences = tokenizer.batch_decode(outputs[:, inputs_len:], skip_special_tokens=True)
                
        del outputs
        torch.cuda.empty_cache()

        # print(generated_sentences)

        # 计算每个句子与图像的匹配度
        # scores = []

        # adding previous sentences to the matching by CLIP
        # generated_sentences.extend(previous_sentences)
        if previous_sentences:
            generated_sentences.append(previous_sentences[-1])
                
        clip_inputs = clip_processor(text=generated_sentences, images=image, return_tensors="pt", padding=True).to(args.device)
                
        # truncation
        if len(clip_inputs['input_ids'][0]) > 77:
            clip_inputs['input_ids'] = clip_inputs['input_ids'][:, :77]
            clip_inputs['attention_mask'] = clip_inputs['attention_mask'][:, :77]
                
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image
        # print(logits_per_image)
                    
        probs = logits_per_image
        # probs = logits_per_image.softmax(dim=1)
        # print(probs)
        # logits_per_image.softmax(dim=-1)

        sentence_prob_pairs = list(zip(generated_sentences, list(probs[0])))

        sorted_sentences = sorted(sentence_prob_pairs, key=lambda x: x[1], reverse=True)

        top5_sentences = [{'sentence': pair[0], 'score': round(float(pair[1]), 3)} for pair in sorted_sentences[:5]]

        print(f'Iteration {polish_ind + 1} candidate top-5:')
        for item in top5_sentences:
            print(f"Score: {item['score']}, Sentence: {item['sentence']}")

        # print(sentence_prob_pairs)

        # 得到匹配度最高的句子
        best_sentence = max(sentence_prob_pairs, key=lambda x: x[1])[0]
        previous_sentences.append(best_sentence.replace('"', '').strip())


        print(f'Iteration {polish_ind + 1} best match:\n{best_sentence}\n\n')

        if len(previous_sentences) >= 3 and previous_sentences[-1] == previous_sentences[-2] and previous_sentences[-2] == previous_sentences[-3]:
            print("Same sentence for three iterations, generation ends.")
            break

    # item_results.append({'attr': desired_attr, 'caption': previous_sentences[-1]})



    print(f'The final result is:\n{previous_sentences[-1]}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lm_type", default='encoder_decoder', type=str)
    parser.add_argument("--lm_path", default='../models/flan-t5-large', type=str)
    parser.add_argument("--clip_path", default='../models/clip-vit-base-patch32', type=str)

    parser.add_argument("--device", default='cuda:5', type=str)



    parser.add_argument("--polish_times", default=100, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--num_return_sequences", default=60, type=int)


    parser.add_argument("--keyword_first", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")



    parser.add_argument("--image_path", default='testbd.jpeg', type=str)

    parser.add_argument("--desired_attr", default='negative', type=str)

    args = parser.parse_args()






    # polish_num = 5
    # desired_attr = "Humorous"
    # desired_attr = "Humorous"


    # 初始化模型和tokenizer
    print("loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path)
    if args.lm_type == 'encoder_decoder':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.lm_path)
    else:     
        model = AutoModelForCausalLM.from_pretrained(args.lm_path)

    if args.lm_type == "decoder_only":
        print(f'model max input tokens {tokenizer.model_max_length} by tokenizer, {model.config.max_position_embeddings} by model')
    # model.half()
    model.to(args.device)

    # 加载CLIP模型和processor
    print("loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(args.clip_path)
    clip_model.to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)



    print("starting generation!")
    with torch.no_grad():
        generate(args, model, tokenizer, clip_model, clip_processor, image_path=args.image_path, desired_attr=args.desired_attr)