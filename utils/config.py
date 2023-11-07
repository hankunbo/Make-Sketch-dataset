import torch
import os
import random
import argparse
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # general arguments
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--img_paths', type=str,default="dataset", help='image file-paths (with wildcards) to process.')
    parser.add_argument('--key_steps', type=int, nargs='+', default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_generation', type=int, default=-1, help='number of sketches to generate, generate all sketches in the dataset if -1')
    parser.add_argument('--chunk', type=int, nargs=2, help='--chunk (num_chunks) (chunk_index)')
    parser.add_argument('--use_gpu', type=bool, default = True , help='--whether using GPU')
    parser.add_argument('--seed', type=int, default = 0 , help='--who the hell knows what its usage')

    parser.add_argument('--enable_color', type=bool, default = True)
    parser.add_argument('--num_strokes', type=int, default = 16)
    parser.add_argument('--num_background', type=int, default = 0)
    parser.add_argument('--num_segments', type=int, default = 1)


    # optimization arguments
    parser.add_argument('--width', type=float, default=1.5, help='foreground-stroke width')
    parser.add_argument('--width_bg', type=float, default=8.0, help='background-stroke width')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1.0)

    # extra arguments
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--no_track_time', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    # CLIPasso arguments
    parser.add_argument("--target", type=str)
    parser.add_argument("--path_svg", type=str, default="none")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--color_vars_threshold", type=float, default=0.0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--control_points_per_seg", type=int, default=4)
    parser.add_argument("--attention_init", type=int, default=1)
    parser.add_argument("--saliency_model", type=str, default="clip")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--xdog_intersec", type=int, default=0)
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)
    parser.add_argument("--percep_loss", type=str, default="none")
    parser.add_argument("--train_with_clip", type=int, default=1)
    parser.add_argument("--clip_weight", type=float, default=0.1)
    parser.add_argument("--start_clip", type=int, default=0)
    parser.add_argument("--num_aug_clip", type=int, default=4)
    parser.add_argument("--include_target_in_aug", type=int, default=0)
    parser.add_argument("--augment_both", type=int, default=1)
    parser.add_argument("--augemntations", type=str, default="affine")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--force_sparse", type=float, default=1)
    parser.add_argument("--clip_conv_loss", type=float, default=1)
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")
    parser.add_argument("--clip_conv_layer_weights", type=str, default="0,0,1.0,1.0,0")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]
    
    args.num_iter = max(args.key_steps)
    args.image_scale = args.image_size
    if args.use_gpu:
        args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")
    args.device = "cuda" if args.use_gpu else "cpu"
    
    args.color_lr = 0.01
    
    return args