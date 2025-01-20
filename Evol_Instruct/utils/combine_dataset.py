import argparse
import os

from Evol_Instruct import client

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', nargs='+', help='The dataset to combine')
    parser.add_argument("--save_name", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.save_name  is None:
        args.save_name = f"mix_{len(args.dataset)}_datasets"
    
    prefix = "s3://syj_test/datasets/medical_train"
    data_dirs = [os.path.join(prefix, name) for name in args.dataset]
    common_files = []
    file_sets = [set(client.listdir(data_dir)) for data_dir in data_dirs]
    # for data_dir in data_dirs:
    #     files_name = set(client.listdir(data_dir))
        
    #     common_files.extend(files_name)
    # print(file_sets)
    common_files = file_sets[0]
    for file_set in file_sets:
        common_files = common_files.intersection(file_set)
        
    
    common_files = list(common_files)
    print(common_files)
    for file in common_files:
        save_name = os.path.join(prefix, args.save_name, file)
        new_data = []
        for data_dir in data_dirs:
            file_path = os.path.join(data_dir, file)
            # print(file_path)
            data = client.read(file_path)
            # client.write(data, save_name, indent=2)
            new_data.extend(data)
        client.write(new_data, save_name, indent=2)
        print(f"Save {file} to {save_name}")