import os
import os.path as osp

import numpy as np
import tensorboardX
import torch

from .mesh import save_obj_file


class TBVisualizer():
    def __init__(self, logdir):
        print("Logging to {}".format(logdir))
        self.log_dir = logdir
        self.stats_dir = f'{self.log_dir}/stats/'
        self.mesh_dir = f'{self.log_dir}/mesh/'
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        if not osp.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)

        self.viz = tensorboardX.SummaryWriter(f'{self.log_dir}')

    def __del__(self):
        self.viz.close()

    def plot_images(self, images, global_step):
        for label, image in images.items():
            assert(np.isfinite(image).all()), label
            if(len(image.shape) == 2):
                dataformats = 'HW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 3):
                dataformats = 'HWC' if (image.shape[2]==3) else 'CHW'
                self.viz.add_image(label,image, global_step, dataformats=dataformats)
            elif(len(image.shape) == 4):
                dataformats = 'NHWC' if (image.shape[3]==3) else 'NCHW'
                self.viz.add_images(label,image, global_step, dataformats=dataformats)
            else:
                raise NotImplementedError


    def plot_videos(self, videos, global_step, fps=4):
        for label, video in videos.items():
            assert(np.isfinite(video).all()), label
            if(len(video.shape) == 4): # t,C,H,W
                assert video.shape[1]==3, f'Invalid video shape:{video.shape}'
                self.viz.add_video(label, video[None], global_step, fps=fps)
            elif(len(image.shape) == 5):
                assert video.shape[2]==3, f'Invalid video shape:{video.shape}'
                self.viz.add_video(label, video, global_step, fps=fps)
            else:
                raise NotImplementedError

    def plot_meshes(self, meshes, global_step):
        for label, mesh in meshes.items():
            vert = mesh['v']
            assert(torch.isfinite(vert).all()), label
            face = mesh['f'] if 'f' in mesh else None
            color = mesh['c'] if 'c' in mesh else None
            config = mesh['cfg'] if 'cfg' in mesh else {}
            self.viz.add_mesh(label,vert,colors=color,faces=face,config_dict=config,global_step=global_step)

    def save_meshes(self, meshes, global_step):
        for label, mesh in meshes.items():
            vert = mesh['v']
            assert(torch.isfinite(vert).all()), label
            face = mesh['f']

            import pymesh
            vert = vert[0] if len(vert.shape)==3 else vert
            face = face[0] if len(face.shape)==3 else face
            outdir = f'{self.mesh_dir}/{label}'
            if not osp.exists(outdir):
                os.makedirs(outdir)
            save_obj_file(f'{outdir}/{global_step}.obj', vert, face)

    def plot_embeddings(self, embeddings, global_step):
        for label, embed in embeddings.items():
            if isinstance(embed,dict):
                mat = embed['mat']
                metadata = embed['metadata'] if 'metadata' in embed else None
                metadata_header = embed['metadata_header'] if 'metadata_header' in embed else None
                label_img = embed['label_img'] if 'label_img' in embed else None
                self.viz.add_embedding(mat,tag=label,global_step=global_step,metadata=metadata, label_img=label_img, metadata_header=metadata_header)
            else:
                assert(torch.isfinite(embed).all()), label
                self.viz.add_embedding(embed,tag=label,global_step=global_step)

    def plot_histograms(self, histograms, global_step):
        for label, hist in histograms.items():
            if isinstance(hist,dict):
                values = hist['values']
                bins = hist['bins'] if 'bins' in hist else 'tensorflow'
                max_bins = hist['max_bins'] if 'max_bins' in hist else None
                self.viz.add_histogram(label,values,global_step=global_step,bins=bins, max_bins=max_bins)
            else:
                assert(torch.isfinite(hist).all()), label
                self.viz.add_histogram(label,hist,global_step=global_step)

    def plot_texts(self, texts, global_step):
        for label, text in texts.items():
            self.viz.add_text(label,text,global_step=global_step)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, global_step, save_meshes=False):
        if 'img' in visuals:
            self.plot_images(visuals['img'], global_step)
        if 'image' in visuals:
            self.plot_images(visuals['image'], global_step)

        if 'video' in visuals:
            fps = visuals['video_fps'] if 'video_fps' in visuals else 4
            self.plot_videos(visuals['video'], global_step, fps=fps)

        if 'mesh' in visuals:
            self.plot_meshes(visuals['mesh'], global_step)
            if save_meshes:
                self.save_meshes(visuals['mesh'], global_step)

        if 'embed' in visuals:
            self.plot_embeddings(visuals['embed'], global_step)

        if 'hist' in visuals:
            self.plot_histograms(visuals['hist'], global_step)

        if 'text' in visuals:
            self.plot_texts(visuals['text'], global_step)

        if 'scalar' in visuals:
            self.plot_current_scalars(global_step, None, visuals['scalar'])

    def save_raw_stats(self, stats, name, epoch):
        path = f'{self.stats_dir}/{name}_{epoch}'
        np.savez(path, **stats)

    def hist_summary_list(self, global_step, tag, data_list):
        t = []
        for l in data_list:
            t.append(l.view(-1))
        t = torch.cat(t)
        self.viz.add_histogram(tag, t.cpu().numpy().reshape(-1), global_step)

    def log_histogram(self, global_step, log_dict):
        for tag, value in log_dict.items():
            self.viz.add_histogram(tag, value.data.cpu().numpy(), global_step)
        return

    def plot_current_scalars(self,global_step, opt, scalars):
        for key, value in scalars.items():
            if isinstance(value, dict):
                self.viz.add_scalars(key, value, global_step)
            else:
                self.viz.add_scalar(key, value, global_step)
