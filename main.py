
import os
import pickle
from torch.utils.data import Dataset, DataLoader, Subset

import utils.config as config
from utils.dataloader import get_dataset
from painterly_rendering import DataGenerator


if __name__ == "__main__":
    args = config.parse_arguments()
    
    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_generator = DataGenerator(args).to(args.device)
    
    path_dicts, mask_areas = data_generator.generate_for_dataset(dataloader, use_tqdm=not args.no_tqdm, track_time=not args.no_track_time)
    
    with open(os.path.join(args.output_dir, f'data_{args.seed}.pkl'), 'wb') as file:
        pickle.dump(path_dicts, file)
    with open(os.path.join(args.output_dir, f'maskareas_{args.seed}.pkl'), 'wb') as file:
        pickle.dump(mask_areas, file)