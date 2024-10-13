import shutil
import random
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Long-tail Distribution Dataset')   
    parser.add_argument('--Imbalance_Factor', type=int,
                        default=10, help='The ratio of the number of samples in the most and the least frequent class. 10/50/100')
    parser.add_argument('--dataset', type=str, default='CIFAR100-C',help='CIFAR10-C,CIFAR-100C')
    parser.add_argument('--source_data_dir', type=str, default='/root/TTA/PCTA-main/code/dataset/CIFAR-100-C/corrupted/severity-5', 
                        help='source data dir')
    parser.add_argument('--target_data_dir', type=str, default='/root/TTA/PCTA-main/code/dataset/CIFAR-100-C-LT', 
                        help='target data dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':    
    args = get_args()
    random.seed(42)

    if args.dataset == 'CIFAR10-C':
        cls_num = 10
        img_max = 1000
    elif args.dataset == 'CIFAR100-C':
        cls_num = 100
        img_max = 100
    else :print("Dataset selected wrong!")

    imbalance_factor=args.Imbalance_Factor
    imb_factor=1/imbalance_factor
    img_num_per_cls = []

    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    
    print("imbalance_factor: ", imbalance_factor)
    print("img_num_per_cls: ", img_num_per_cls)

    source_base_dir = Path(args.source_data_dir)
    target_base_dir = Path(args.target_data_dir)
    target_base_dir.mkdir(parents=True, exist_ok=True)

    for noise_type in ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", 
                        "defocus_blur", "brightness", "fog", "zoom_blur", "frost", 
                        "glass_blur", "impulse_noise", "contrast", "jpeg_compression", 
                        "elastic_transform"]:
    
        source_type_dir = source_base_dir / noise_type
        target_type_dir = target_base_dir / noise_type
        
        target_type_dir.mkdir(parents=True, exist_ok=True)
        
        for cls_idx in range(cls_num):
            cls_folder = source_type_dir / f"{cls_idx}"
            target_cls_folder = target_type_dir / f"{cls_idx}"
            target_cls_folder.mkdir(parents=True, exist_ok=True)
            img_files = list(cls_folder.glob("*"))
            img_files = [file for file in img_files if '.ipynb_checkpoints' not in file.as_posix()]
            num_imgs_to_select = img_num_per_cls[cls_idx]
            if len(img_files) > 0:
                selected_imgs = random.sample(img_files, min(num_imgs_to_select, len(img_files)))
                for img_file in selected_imgs:
                    shutil.copy(img_file, target_cls_folder / img_file.name)
            else:
                print(f"Warning: There are no image files in the {cls_folder} directory.")
