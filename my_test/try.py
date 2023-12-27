import torch
import json

def check_inference_predicts(data_path):
    data = torch.load(data_path)
    print(data)
    # print(data['ret_scores'][958:1000], data['ret_bboxes'].shape) 
    binary_scores = torch.where(
        data['ret_scores'] > 0.3, 
        1, 0 
    )

    print("indices: ", torch.nonzero(binary_scores))
    
def check_query_set_number(json_path):
    with open (json_path, 'r') as f:
        data = json.load(f)

    total_num = 0
    for key, value in data.items():
        for set in value['predictions']:
            num = len(set['query_sets'])
            total_num += num
    print(total_num)

if __name__ == "__main__":
    annot_key = "0b736368-e31c-40b6-83e9-069c434e47db_1"
    data_path = f"/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/VQLoC/output/ego4d_vq2d/val/validate/inference_cache_val/{annot_key}.pt"
    data_path = "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/VQLoC/output/ego4d_vq2d/eval/eval/inference_cache_eval/0b7033cd-2029-48c1-aa8d-216481d65802_1.pt"
    # check_inference_predicts(data_path)
    json_path = "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/VQLoC/output/ego4d_vq2d/val/validate/inference_cache_val_results.json"
    check_query_set_number(json_path)