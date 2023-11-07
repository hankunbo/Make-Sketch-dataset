import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from third_party.clipasso.models.painter_params import Painter, PainterOptimizer
from third_party.clipasso.models.loss import Loss
from third_party.clipasso import sketch_utils as utils

# The main class that generates sketch data with the specified image dataset.
class DataGenerator(nn.Module):
    def __init__(self, args):
        super(DataGenerator, self).__init__()
        
        self.args = args
        
        # Initialize the renderers.
        renderers = []
        renderers.append(Painter(
            args,
            args.num_strokes, args.num_segments,
            imsize=args.image_scale,
            device=args.device,
        ))
        for _ in range(args.batch_size - 1):
            renderers.append(Painter(
                args,
                args.num_strokes, args.num_segments,
                imsize=args.image_scale,
                device=args.device,
                clip_model=renderers[0].clip_model,
                clip_preprocess=renderers[0].clip_preprocess,
                dino_model=renderers[0].dino_model
            ))

        self.renderers = nn.ModuleList(renderers)
        self.criterion = Loss(args)
        self.u2net = utils.get_u2net(args)

    def save_sample_visualization(self, sample_name, sample_image, sample_paths):
        fig, axs = plt.subplots(1, len(sample_paths)+1, figsize=(3+3*len(sample_paths), 3))

        axs[0].set_title('image')
        axs[0].imshow(sample_image)

        t = np.linspace(0, 1, 10)
        for i, (step, step_paths) in enumerate(sample_paths.items()):
            curves = cubic_bezier(step_paths['pos'], t)
            axs[i+1].set_title(f'step {step}')
            for curve, color, width in zip(curves, step_paths['color'], step_paths['radius']):
                axs[i+1].plot(*curve.T[::-1], c=color)
            axs[i+1].set_ylim(1,-1)
            axs[i+1].tick_params(axis='both', which='major', labelsize=5)

        vis_filename = os.path.join(self.args.output_dir, f'vis/{sample_name}.jpg')
        fig.savefig(vis_filename)
        plt.close()

    def _generate(self, image, mask, num_iter, num_strokes, width, attn_colors, path_dicts=None, gradual_colors=True, use_tqdm=False):
        curr_batch_size = image.size(0)
        if path_dicts is None:
            path_dicts = [None] * curr_batch_size
        renderers = self.renderers[:curr_batch_size]

        for renderer, curr_image, curr_mask, path_dict in zip(renderers, image, mask, path_dicts):
            renderer.set_random_noise(0)
            renderer.init_image(
                target_im=curr_image.unsqueeze(0),
                mask=curr_mask.unsqueeze(0),
                stage=0,
                randomize_colors=False,
                attn_colors=attn_colors,
                attn_colors_stroke_sigma=5.0,
                path_dict=path_dict,
                new_num_strokes=num_strokes,
                new_width=width
            )

        if num_iter == 0:
            for renderer in renderers:
                for key_step in self.args.key_steps:
                    renderer.log_shapes(str(key_step))
            path_dicts = [renderer.path_dict_np(radius=width) for renderer in renderers]
            if gradual_colors:
                for sample_paths in path_dicts:
                    ts = np.linspace(0, 1, len(sample_paths))
                    for t, step_paths in zip(ts, sample_paths.values()):
                        step_paths['color'] *= t
            return path_dicts

        optimizer = PainterOptimizer(self.args, renderers)
        optimizer.init_optimizers()

        steps = range(num_iter)
        if use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            for renderer in renderers:
                renderer.set_random_noise(step)

            optimizer.zero_grad_()
            sketches = torch.cat([renderer.get_image().to(self.args.device) for renderer in renderers], dim=0)
            #print(list(self.criterion(sketches, image.detach(), step, points_optim=optimizer).values())[:-2])
            loss = sum(list(self.criterion(sketches, image.detach(), step, points_optim=optimizer).values())[:-2])
            loss.backward()
            optimizer.step_(optimize_points=True, optimize_colors=False)

            if (step+1) in self.args.key_steps:
                for renderer in renderers:
                    renderer.log_shapes()
                    renderer.log_shapes(str(step+1))

        return [renderer.path_dict_np(radius=width) for renderer in renderers]

    def generate_for_batch(self, index, image, use_tqdm=False):
        sample_names = [f'{idx}_{self.args.seed}' for idx in index.tolist()]

        foreground, background, mask, _ = utils.get_mask_u2net_batch(self.args, image, net=self.u2net, return_background=True)
        with torch.no_grad():
            mask_areas = mask.view(mask.size(0), -1).mean(dim=1).tolist()
            mask_areas = dict(zip(sample_names, mask_areas))
        
        num_strokes_fg = self.args.num_strokes - self.args.num_background
        num_strokes_bg = self.args.num_background
        stroke_width_fg = self.args.width
        stroke_width_bg = self.args.width_bg

        path_dicts = self._generate(foreground, mask, self.args.num_iter, num_strokes_fg, stroke_width_fg, False, use_tqdm=use_tqdm)
        
        if not self.args.enable_color:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas

        color_dicts = self._generate(foreground, mask, 0, None, stroke_width_fg, True, path_dicts=path_dicts, use_tqdm=use_tqdm)
        for paths, colors in zip(path_dicts, color_dicts):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['color'] = colors[step]['color']

        if num_strokes_bg <= 0:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas
        
        path_dicts_bg = self._generate(background, 1 - mask, 0, num_strokes_bg, stroke_width_bg, True, use_tqdm=use_tqdm)
        for paths, paths_bg in zip(path_dicts, path_dicts_bg):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['pos'] = np.concatenate([paths[step]['pos'], paths_bg[step]['pos']], axis=0)
                paths[step]['color'] = np.concatenate([paths[step]['color'], paths_bg[step]['color']], axis=0)
                if 'radius' in paths[step]:
                    paths[step]['radius'] = np.concatenate([paths[step]['radius'], paths_bg[step]['radius']], axis=0)

        path_dicts = dict(zip(sample_names, path_dicts))
        return path_dicts, mask_areas
    
    def generate_for_dataset(self, dataloader, use_tqdm=False, track_time=False):
        path_dicts = {}
        mask_areas = {}

        if track_time:
            start_time = time.time()

        min_index = next(iter(dataloader.dataset.indices))
        max_index = next(iter(reversed(dataloader.dataset.indices)))

        generated_samples = 0
        total_samples = len(dataloader.dataset)
        
        for index, image in dataloader:
            if track_time:
                print(f'generating samples for {index.min().item()}..{index.max().item()} of {min_index}..{max_index}:')
            image = image.to(self.args.device)
            
            batch_path_dicts, batch_mask_areas = self.generate_for_batch(index, image, use_tqdm=use_tqdm)
            path_dicts.update(batch_path_dicts)
            mask_areas.update(batch_mask_areas)

            if self.args.visualize:
                for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
                    sample_image = image[i].detach().cpu().permute(1, 2, 0).numpy()
                    self.save_sample_visualization(sample_name, sample_image, sample_paths)

            if track_time:
                generated_samples += image.size(0)
                completion = generated_samples / total_samples
                time_passed = time.time() - start_time
                time_left = time_passed / completion - time_passed
                tp_days, tp_hours, tp_minutes, _ = to_dhms(time_passed)
                tl_days, tl_hours, tl_minutes, _ = to_dhms(time_left)
                print(
                    f'{completion*100:.02f}% ({generated_samples}/{total_samples}) complete.'\
                    f' {tp_days}d {tp_hours}h {tp_minutes}m passed.'\
                    f' expected {tl_days}d {tl_hours}h {tl_minutes}m left.'
                )

        if track_time:
            print(f'took {time_passed:.02f}s to generate {total_samples} samples.')

        return path_dicts, mask_areas

def cubic_bezier(p, t):
    p = p.reshape(-1, 4, 1, 2)
    t = t.reshape(1, -1, 1)
    return ((1-t)**3)*p[:,0] + 3*((1-t)**2)*t*p[:,1] + 3*(1-t)*(t**2)*p[:,2] + (t**3)*p[:,3]

def to_dhms(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds
