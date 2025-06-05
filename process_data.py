import json
from tqdm import tqdm
import argparse
import os
import re
import random


# process flickr30k, senticap, and coco dataset
# return type: {'train': {'normal': [{'img_id': , ''}, ...], 'positive': [], 'negative': []}, 'test': ...}
def processing_data(caption_dataset_path, data_type='flickr30k'):

    with open(caption_dataset_path, 'r') as f:
        data = json.load(f)

    processed_data = {'train': {}, 'val': {}, 'test': {}}
    
    if data_type == 'flickr30k':

        # initialize
        processed_data['train']['normal'] = []
        processed_data['val']['normal'] = []
        processed_data['test']['normal'] = []

        for item in tqdm(data['images']):
            # record image id, filename, and raw sentence
            processed_item = {'img_id': item['imgid'], 'filename': item['filename'], 'sentences': [sent['raw'] for sent in item['sentences']]}

            processed_data[item['split']]['normal'].append(processed_item)

            # if item['split'] == 'train':
            #     processed_data['train']['normal'].append(processed_item)
            # elif item['split'] == 'test':
            #     processed_data['test']['normal'].append(processed_item)
        
        return processed_data

    elif data_type == 'senticap':

        # initialize
        processed_data['train']['positive'] = []
        processed_data['train']['negative'] = []
        processed_data['val']['positive'] = []
        processed_data['val']['negative'] = []
        processed_data['test']['positive'] = []
        processed_data['test']['negative'] = []

        for item in tqdm(data['images']):

            positive_sentences = []
            negative_sentences = []

            # 0->negative, 1->positive
            for sent in item['sentences']:
                if sent['sentiment'] == 1:
                    positive_sentences.append(sent['raw'])
                elif sent['sentiment'] == 0:
                    negative_sentences.append(sent['raw'])

            # record image id, filename, and raw sentence
            processed_item_positive = {'img_id': item['imgid'], 'filename': item['filename'], 'sentences': positive_sentences}
            processed_item_negative = {'img_id': item['imgid'], 'filename': item['filename'], 'sentences': negative_sentences}

            if(processed_item_positive['sentences'] != []):
                processed_data[item['split']]['positive'].append(processed_item_positive)
            
            if(processed_item_negative['sentences'] != []):
                processed_data[item['split']]['negative'].append(processed_item_negative)

            # if item['split'] == 'train':
            #     if(processed_item_positive != []):
            #         processed_data['train']['positive'].append(processed_item_positive)
            #     processed_data['train']['negative'].append(processed_item_negative)
            # elif item['split'] == 'test':
            #     processed_data['test']['positive'].append(processed_item_positive)
            #     processed_data['test']['negative'].append(processed_item_negative)

        return processed_data

    elif data_type == 'coco':
        
        # initialize
        processed_data['restval'] = {'normal': []}
        processed_data['train']['normal'] = []
        processed_data['val']['normal'] = []
        processed_data['test']['normal'] = []

        for item in tqdm(data['images']):
            # record image id, filename, and raw sentence
            processed_item = {'img_id': item['imgid'], 'filename': os.path.join(item['filepath'], item['filename']), 'sentences': [sent['raw'] for sent in item['sentences']]}

            processed_data[item['split']]['normal'].append(processed_item)
        
        return processed_data
    
    else:
        raise NotImplementedError


# process flickrstyle10k dataset
def processing_flickrstyle10k(caption_dataset_folder_path):

    if os.path.isfile(caption_dataset_folder_path):
        raise NotADirectoryError('caption_dataset_folder_path while processing flickrstyle10k dataset should be a folder path, not a file path')

    random_order = list(range(7000))

    # shuffle the list
    random.shuffle(random_order)



    # processed_data = {'train': {}}
    processed_data = {'train': {}, 'test': {}}

    # initialize
    processed_data['train']['humor'] = []
    processed_data['train']['romantic'] = []

    processed_data['test']['humor'] = []
    processed_data['test']['romantic'] = []

    with open(os.path.join(caption_dataset_folder_path, 'humor/train.p'), 'r') as filename_f:
        with open(os.path.join(caption_dataset_folder_path, 'humor/funny_train.txt'), 'r') as sent_f:
            file_names = filename_f.readlines()
            sentences = sent_f.readlines()
                
            ind = 0
            with tqdm(total=len(file_names)//2-1) as pbar:
                # ind is the row number in train.p representing the row number in funny_train.txt
                # ind + 1 is the file name of the image
                while ind < len(file_names):
                    # find row number in filename
                    # print(file_names[ind])
                    row_num = int(''.join(re.findall(r'\d+', file_names[ind])))
                    # print(row_num)

                    # 7000 is the number of captions in the public part of flickrstyle10k dataset
                    if(row_num > 7000):
                        ind += 2
                        continue

                    # remove 'V' or 'aV' and strip '\n' in filename
                    # strip '\n' in sentence
                    humor_item = {'img_id': row_num - 1, 'filename': re.sub(r'^a?V', '', file_names[ind + 1]).strip(), 'sentences': [sentences[row_num - 1].strip()]}

                    if random_order[row_num - 1] < 6000:
                        processed_data['train']['humor'].append(humor_item)
                    else:
                        processed_data['test']['humor'].append(humor_item)

                    # first row is row number, second row is image name
                    ind += 2

                    pbar.update(1)

    # return processed_data

    with open(os.path.join(caption_dataset_folder_path, 'romantic/train.p'), 'r') as filename_f:
        with open(os.path.join(caption_dataset_folder_path, 'romantic/romantic_train.txt'), 'r', errors='replace') as sent_f:
            file_names = filename_f.readlines()
            sentences = sent_f.readlines()
                
            ind = 0

            with tqdm(total=len(file_names)//2-1) as pbar:
                # ind is the row number in train.p representing the row number in romantic_train.txt
                # ind + 1 is the file name of the image
                while ind < len(file_names):
                    # find row number in filename
                    row_num = int(''.join(re.findall(r'\d+', file_names[ind])))

                    # 7000 is the number of captions in the public part of flickrstyle10k dataset
                    if(row_num > 7000):
                        ind += 2
                        continue

                    # remove 'V' or 'aV' and strip '\n' in filename
                    # strip '\n' in sentence
                    romantic_item = {'img_id': row_num - 1, 'filename': re.sub(r'^a?V', '', file_names[ind + 1]).strip(), 'sentences': [sentences[row_num - 1].strip()]}


                    if random_order[row_num - 1] < 6000:
                        processed_data['train']['romantic'].append(romantic_item)
                    else:
                        processed_data['test']['romantic'].append(romantic_item)

                    # first row is row number, second row is image name
                    ind += 2

                    pbar.update(1)


    
    # checking the processed data
    print("checking the processed data...")
    for ind, item in enumerate(processed_data['train']['humor']):
        if item['img_id'] != 0 and item['img_id'] != processed_data['train']['humor'][ind - 1]['img_id'] + 1:
            print(f'error detected in humor dataset: {item["img_id"]}')
            # print(item)
        # print(item)

    for ind, item in enumerate(processed_data['train']['romantic']):
        if item['img_id'] != 0 and item['img_id'] != processed_data['train']['romantic'][ind - 1]['img_id'] + 1:
            print(f'error detected in romantic dataset: {item["img_id"]}')
        # print(item)

    return processed_data


# process flickrstyle10k dataset
def processing_flickrstyle10k_by_zgsplit(caption_dataset_folder_path, zerogen_humor_dataset_path, zerogen_romantic_dataset_path):

    if os.path.isfile(caption_dataset_folder_path):
        raise NotADirectoryError('caption_dataset_folder_path while processing flickrstyle10k dataset should be a folder path, not a file path')





    # processed_data = {'train': {}}
    processed_data = {'train': {}, 'test': {}}

    # initialize
    processed_data['train']['humor'] = []
    processed_data['train']['romantic'] = []

    processed_data['test']['humor'] = []
    processed_data['test']['romantic'] = []

    zg_set = {'humor': set(), 'romantic': set()}

    with open(zerogen_humor_dataset_path, 'r') as f:
        zg_humor_data = json.load(f)
        for item in zg_humor_data:
            zg_set['humor'].add(item['image_name'].replace('.jpg', ''))

    with open(zerogen_romantic_dataset_path, 'r') as f:
        zg_romantic_data = json.load(f)
        for item in zg_romantic_data:
            zg_set['romantic'].add(item['image_name'].replace('.jpg', ''))

    # print(zg_set)


    with open(os.path.join(caption_dataset_folder_path, 'humor/train.p'), 'r') as filename_f:
        with open(os.path.join(caption_dataset_folder_path, 'humor/funny_train.txt'), 'r') as sent_f:
            file_names = filename_f.readlines()
            sentences = sent_f.readlines()
                
            ind = 0
            with tqdm(total=len(file_names)//2-1) as pbar:
                # ind is the row number in train.p representing the row number in funny_train.txt
                # ind + 1 is the file name of the image
                while ind < len(file_names):
                    # find row number in filename
                    # print(file_names[ind])
                    row_num = int(''.join(re.findall(r'\d+', file_names[ind])))
                    # print(row_num)

                    # 7000 is the number of captions in the public part of flickrstyle10k dataset
                    if(row_num > 7000):
                        ind += 2
                        continue

                    

                    # remove 'V' or 'aV' and strip '\n' in filename
                    filename = re.sub(r'^a?V', '', file_names[ind + 1]).strip()

                    # strip '\n' in sentence
                    humor_item = {'img_id': row_num - 1, 'filename': filename, 'sentences': [sentences[row_num - 1].strip()]}

                    
                    if filename.split("_")[0] in zg_set['humor']:
                        processed_data['test']['humor'].append(humor_item)
                    else:
                        processed_data['train']['humor'].append(humor_item)

                    # first row is row number, second row is image name
                    ind += 2

                    pbar.update(1)

    # return processed_data

    with open(os.path.join(caption_dataset_folder_path, 'romantic/train.p'), 'r') as filename_f:
        with open(os.path.join(caption_dataset_folder_path, 'romantic/romantic_train.txt'), 'r', errors='replace') as sent_f:
            file_names = filename_f.readlines()
            sentences = sent_f.readlines()
                
            ind = 0

            with tqdm(total=len(file_names)//2-1) as pbar:
                # ind is the row number in train.p representing the row number in romantic_train.txt
                # ind + 1 is the file name of the image
                while ind < len(file_names):
                    # find row number in filename
                    row_num = int(''.join(re.findall(r'\d+', file_names[ind])))

                    # 7000 is the number of captions in the public part of flickrstyle10k dataset
                    if(row_num > 7000):
                        ind += 2
                        continue

                    # remove 'V' or 'aV' and strip '\n' in filename
                    filename = re.sub(r'^a?V', '', file_names[ind + 1]).strip()

                    # strip '\n' in sentence
                    romantic_item = {'img_id': row_num - 1, 'filename': filename, 'sentences': [sentences[row_num - 1].strip()]}

                    # if random_order[row_num - 1] < 6000:
                    #     processed_data['train']['romantic'].append(romantic_item)
                    # else:
                    #     processed_data['test']['romantic'].append(romantic_item)

                    if filename.split("_")[0] in zg_set['romantic']:
                        processed_data['test']['romantic'].append(romantic_item)
                    else:
                        processed_data['train']['romantic'].append(romantic_item)

                    # first row is row number, second row is image name
                    ind += 2

                    pbar.update(1)


    print(f'processed_data: train test\n     humor       {len(processed_data["train"]["humor"])} {len(processed_data["test"]["humor"])}\n   romantic      {len(processed_data["train"]["romantic"])} {len(processed_data["test"]["romantic"])}')


    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_dataset_path", default='autodl-tmp/data/', type=str)
    parser.add_argument("--zerogen_dataset_path", default='/Users/shi/Downloads/extra_data/flickr10_data_zerogen', type=str)
    parser.add_argument("--data_type", default='flickr30k', type=str)
    parser.add_argument("--output_path", default='processed_filckr30k.json', type=str)
    args = parser.parse_args()

    if args.data_type == 'flickrstyle10k':
        # processed_data = processing_flickrstyle10k(args.caption_dataset_path)
        processed_data = processing_flickrstyle10k_by_zgsplit(args.caption_dataset_path, os.path.join(args.zerogen_dataset_path, 'humor/humor_caption_test.json'), os.path.join(args.zerogen_dataset_path, 'romantic/romantic_caption_test.json'))
    else:
        processed_data = processing_data(args.caption_dataset_path, args.data_type)

    with open(args.output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

