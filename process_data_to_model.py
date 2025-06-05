import json
import argparse
import os

def process_data_for_capdec(args, attrs, data):
    for attr in attrs:
        for split in ['train', 'test']:
            new_data = []
            for item in data[split][attr]:
                new_item = {}
                new_item['image_id'] = item['img_id']
                new_item['filename'] = item['filename']
                new_item['caption'] = item['sentences'][0]
                new_data.append(new_item)

            with open(os.path.join(args.output_folder_path, f'capdec_{args.dataset_type}_{attr}_{split}.json'), 'w') as file:
                json.dump(new_data, file, indent=4)

                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--proc_data_path", default='dataset/processed_dataset/processed_flickrstyle10k_zgsplit.json', type=str)
    parser.add_argument("--dataset_type", default='flickrstyle10k', type=str)
    parser.add_argument("--model_type", default='capdec', type=str)
    parser.add_argument("--output_folder_path", default='dataset/processed_dataset_for_model2', type=str)

    args = parser.parse_args()

    # process attribute for dataset type
    if args.dataset_type == 'flickrstyle10k':
        attrs = ['humor', 'romantic']
    elif args.dataset_type == 'senticap':
        attrs = ['negative', 'positive']
    else:
        attrs = ['normal']

    with open(args.proc_data_path, "r") as file:
        data = json.load(file)

    if args.model_type == 'capdec':
        process_data_for_capdec(args, attrs, data)
    elif args.model_type == 'other_model':
        raise NotImplementedError
        # process_data_for_other_model(args, data)
    else:
        raise NotImplementedError
    


