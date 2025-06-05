# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import json
import argparse
import os
import csv

def calculate_scores(reference_sentences, generated_sentences):

    # bleu1 = 0
    # bleu4 = 0


    # BLEU scores
    bleu_scorer = Bleu()
    bleu_scores, _ = bleu_scorer.compute_score(reference_sentences, generated_sentences)
    # print(bleu_scores)
    bleu1 = bleu_scores[0]
    bleu3 = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2]) / 3
    bleu4 = sum(bleu_scores) / len(bleu_scores)

    # METEOR score
    # meteor_scorer = Meteor()
    # meteor_score, _ = meteor_scorer.compute_score(reference_sentences, generated_sentences)
    # meteor = meteor_score[0]
    meteor = 1


    # ROUGE-L score
    # rouge = Rouge()
    # scores = rouge.get_scores(generated_sentence, reference_sentences[0])
    # rouge_l = scores[0]['rouge-l']['f']

    rouge = Rouge()
    rouge_l_scores, _ = rouge.compute_score(reference_sentences, generated_sentences)
    rouge_l = rouge_l_scores



    # CIDEr and SPICE scores
    cider = Cider()
    scores = cider.compute_score(reference_sentences, generated_sentences)
    cider_score = scores[0]
    # scores = spice.compute_score({0: reference_sentences}, {0: [generated_sentence]})


    # spice = Spice()
    # scores = spice.compute_score(reference_sentences, generated_sentences)
    # spice_score = scores[0]
    spice_score = 1

    return bleu1, bleu_scores[2], bleu_scores[3], bleu3, bleu4, meteor, rouge_l, cider_score, spice_score


def read_json(output_path):
    with open(output_path, "r", errors='ignore') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("--folder_path", default='/Users/shi/Downloads/cic_result', type=str)
    parser.add_argument("--ground_truth_path", default='dataset/processed_dataset/processed_senticap.json', type=str)
    # parser.add_argument("--eval_result_path", default='eval_result.csv', type=str)


    parser.add_argument("--folder_path", default='/Users/shi/Downloads/cic_old_result/no_recording_previous', type=str)
    parser.add_argument("--eval_result_path", default='eval_result/senticap_eval_result.csv', type=str)
    parser.add_argument("--polish_time", default=10, type=int)

    parser.add_argument("--no_recording_previous", action="store_true")

    parser.add_argument("--zerogen", action="store_true")

    parser.add_argument("--dataset_split", default='test', type=str)

    args = parser.parse_args()

    eval_results = []


    # processing ground truth data
    gt_data = read_json(args.ground_truth_path)[args.dataset_split]

    gt_map = {}

    for attr, gts in gt_data.items():
        gt_map[attr] = {}
        for gt in gts:
            if args.zerogen:
                new_filename = gt['filename'].split('_')[0] + '.jpg'
                # print(new_filename)
                gt_map[attr][new_filename] = gt['sentences']
            else:
                gt_map[attr][gt['filename']] = gt['sentences']



    # processing generation data
    for root, dirs, files in os.walk(args.folder_path):
        for file in files:
            data_path = os.path.join(root, file)
            if data_path[-5: ] != '.json':
                continue

            print(data_path)
            data = read_json(data_path)

            

            for attr, generations in data.items():

                if args.no_recording_previous:
                    generated_sentence_by_polish_time = {100: {}}
                else:
                    # generated_sentence_by_polish_time = {2: {}, -1: {}}

                    generated_sentence_by_polish_time = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 0: {}}
                reference_sentences = {}


                for ind, generation in enumerate(generations):
                    image_name = generation['image_name']

                    reference_sentences[ind] = gt_map[attr][image_name]

                    if args.no_recording_previous:
                        generated_sentence_by_polish_time[100][ind] = [generation['captions']]
                    else:

                        for polish_time in generated_sentence_by_polish_time.keys():
                            # if polish_time >= len(generation['captions']):
                            #     # generated_sentence_by_polish_time.pop(polish_time)
                            #     continue
                            sentence = generation['captions'][polish_time - 1]
                            generated_sentence_by_polish_time[polish_time][ind] = [sentence]


                    # generated_sentence_1
                        

                    
                # for polish_time, generated_sentences in generated_sentence_by_polish_time.items():
                #     bleu1, bleu4, meteor, rouge_l, cider_score, spice_score = calculate_scores(reference_sentences, generated_sentences)
                #     # Create a list to store the results
                #     results = []

                # Iterate over the generated_sentence_by_polish_time dictionary
                for polish_time, generated_sentences in generated_sentence_by_polish_time.items():
                    if not generated_sentences:
                        continue
                    # Calculate the scores for each polish time
                    # print(generated_sentences)
                    bleu1, bleu_o3, bleu_o4, bleu3, bleu4, meteor, rouge_l, cider_score, spice_score = calculate_scores(reference_sentences, generated_sentences) 
                    print(bleu1, bleu_o3, bleu_o4, bleu3, bleu4, meteor, rouge_l, cider_score, spice_score)
                    # Append the scores to the results list
                    eval_results.append([data_path, attr, polish_time, bleu1, bleu_o3, bleu_o4, bleu3, bleu4, meteor, rouge_l, cider_score, spice_score])



                    

                  
            


    # Write the results to the CSV file
    with open(args.eval_result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Data Path', 'Attribute', 'Polish Time', 'BLEU-1', 'BLEU-only3', 'BLEU-only4', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE'])
        writer.writerows(eval_results)

    # reference_sentences = ['this is a train sentence', 'it is a sentence for testing', 'a test sentence is that']
    # generated_sentence = 'this is a test sentence'
    
    # print('BLEU-1:', bleu1)
    # print('BLEU-4:', bleu4)
    # print('METEOR:', meteor)
    # print('ROUGE-L:', rouge_l)
    # print('CIDEr:', cider_score)
    # print('SPICE:', spice_score)
