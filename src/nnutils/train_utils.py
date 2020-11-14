"""Generic Training Utils.
"""

from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import time
from collections import Counter

import matplotlib.pyplot as plt
import torch
from absl import flags
from tqdm import tqdm

from ..optim import adam as ams_adam
from ..utils.misc import AverageMeter
from ..utils.tb_visualizer import TBVisualizer

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../../', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'Number of epochs for which it was been pretrained')
flags.DEFINE_string('pretrained_epoch_label', '', 'If not empty, we will pretain from an existing saved model.')
flags.DEFINE_string('pretrained_network_dir', '', 'If empty, will use "name".')
flags.DEFINE_string('pretrained_network_path', '', 'If empty, will use "cache_dir/name".')

flags.DEFINE_string('pretrained_camera_params_path', '', 'Path to pretrained camera parameters')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')
flags.DEFINE_float('max_grad_norm', 100, 'Clipping gradient norm at this value')

flags.DEFINE_integer('batch_size', 12, 'Size of minibatches')
flags.DEFINE_integer('num_iter', 1000000, 'Number of training iterations. 0 -> Use epoch_iter')
flags.DEFINE_integer('start_num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_boolean('shuffle_data', True, 'Whether dataloader shuffles data')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 20000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 2, 'save model every k epochs')

## Flags for visualization
flags.DEFINE_integer('display_freq', 500, 'visuals logging frequency')
flags.DEFINE_integer('display_freq_init', 50, 'visuals logging frequency (initially)')
flags.DEFINE_integer('display_init_iter', 100, 'these initial iters log more frequently at display_freq_init')
flags.DEFINE_boolean('display_visuals', True, 'whether to display images')
flags.DEFINE_boolean('print_scalars', False, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', True, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_boolean('save_meshes', True, 'Save meshes to file also')
flags.DEFINE_integer('break_epoch_early', -1, 'break epoch early after these iterations')
flags.DEFINE_integer('visualize_camera_freq', 1, 'Visualize camera histogram every k epochs')
flags.DEFINE_boolean('save_camera_pose_dict', False, 'Save camera pose dict for entire dataset to file')
flags.DEFINE_boolean('save_partial_camera_pose_dict', False, 'Save partial camera pose dict')

#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        self.train_mode = opts.is_train
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))
        self.config_str = '<pre>' + ''.join([f'{k:30}:{opts.__getattr__(k)}\n' for k in dir(opts)]) + '</pre>'
        self.total_loss = torch.tensor(0, dtype=torch.float32)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        if network is not None:
            save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(network.state_dict(), save_path)
            if torch.cuda.is_available():
                network.cuda()
            return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir='', path='', key_prefix='', strict=False):
        if path == '':
            save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
            if network_dir == '':
                network_dir = self.save_dir
            path = os.path.join(network_dir, save_filename)

        assert(os.path.isfile(path))
        model_state = torch.load(path)
        new_model_state = {}
        for key in model_state.keys():
            if key.startswith(key_prefix):
                new_model_state[key[len(key_prefix):]] = model_state[key]

        incompat_keys = network.load_state_dict(new_model_state, strict=strict)
        print('Loading weights from {}'.format(path))
        print(incompat_keys)
        return

    def homogenize_coordinates(self, locations):
        '''
        :param locations: N x 3 array of locations
        :return: homogeneous coordinates
        '''
        ones = torch.ones((locations.size(0), 1))
        homogenous_location = torch.cat([locations, ones], dim=1)
        return homogenous_location

    def count_number_pairs(self, rois):
        counts = Counter(rois[:,0].numpy().tolist())
        pairs = sum([v*v for (k,v) in counts.items() if v > 1])
        return pairs

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_parameters(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def reset_epoch_statistics(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_epoch_statistics(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_param_groups(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_camera_param_groups(self):
        '''Should be implemented by the child class.'''
        return [{'params':torch.nn.Parameter(torch.tensor([0.])),'lr':0}]

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=opts.learning_rate)
        self.optimizer = ams_adam.Adam(
            self.get_param_groups(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))

        if opts.optimizeAlgo=='sgd':
            self.camera_optimizer = torch.optim.SGD(
                self.get_camera_param_groups())
        elif opts.optimizeAlgo=='adam':
            self.camera_optimizer = ams_adam.Adam_NonZeroGradOnly(
                self.get_camera_param_groups(), betas=(opts.beta1, 0.999))
        else:
            raise ValueError

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        return torch.tensor(ave_grads)


    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.tb_visualizer = TBVisualizer(osp.join(opts.cache_dir, 'logs', opts.name))
        tb_visualizer = self.tb_visualizer
        tb_visualizer.viz.add_text('config',self.config_str)

        dataset_size = len(self.dataloader)

        self.real_iter = opts.start_num_iter
        self.epochs_done = float(self.real_iter) / dataset_size
        if not (abs(self.epochs_done-opts.num_pretrain_epochs) <= 1):
            print('#############################################')
            print(f'##### Warning: epoch_done ({self.epochs_done}) doesn\'t match num_pretrain_epochs ({opts.num_pretrain_epochs})')
            print('#############################################')
        # assert(abs(self.epochs_done-opts.num_pretrain_epochs) <= 1)

        total_steps = opts.start_num_iter
        start_time = time.time()
        opts.save_epoch_freq = min(opts.save_epoch_freq, opts.num_epochs - opts.num_pretrain_epochs)
        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            epoch_iter = 0
            loss_meter = AverageMeter()
            self.reset_epoch_statistics()

            if (opts.visualize_camera_freq>0) and (((epoch+1)%opts.visualize_camera_freq)==0):
                visualize_camera_dist = True
            else:
                visualize_camera_dist = False

            backward_camera = opts.optimizeCameraCont and (opts.optimizeLR>0)

            pbar = tqdm(self.dataloader,dynamic_ncols=True,total=dataset_size,desc=f'e{epoch}')
            for i, batch in enumerate(pbar):
                if (i>=opts.break_epoch_early) and (opts.break_epoch_early>=0):
                    break
                iter_start_time = time.time()
                grad_norm = 0
                self.set_input(batch)
                if not self.invalid_batch:
                    self.real_iter +=1
                    self.epochs_done = float(self.real_iter) / dataset_size
                    self.optimizer.zero_grad()
                    # self.camera_optimizer.zero_grad()
                    self.forward(visualize_camera_dist=visualize_camera_dist)
                    self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.total_loss.item()
                    if self.train_mode:
                        self.total_loss.backward(retain_graph=backward_camera)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),opts.max_grad_norm)
                        self.optimizer.step()

                        if backward_camera:
                            self.camera_optimizer.zero_grad()
                            self.camera_update_loss.backward()
                            self.camera_optimizer.step()

                total_steps += 1
                epoch_iter += 1
                loss_meter.append(self.total_loss.item())
                pbar.set_postfix(iter=self.real_iter, loss_avg=loss_meter.average(), loss_recent=loss_meter.recent_average())


                if opts.display_visuals and ((total_steps % opts.display_freq == 0) or
                                            ((total_steps<=opts.display_init_iter) and
                                                (total_steps % opts.display_freq_init == 0))):
                    tb_visualizer.display_current_results(self.get_current_visuals(), total_steps, save_meshes=opts.save_meshes)


                if (opts.print_scalars or opts.plot_scalars) and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    scalars['grad_norm'] = grad_norm
                    if opts.plot_scalars:
                        tb_visualizer.plot_current_scalars(total_steps, opts, scalars)


                if self.train_mode and (total_steps % opts.save_latest_freq == 0):
                    print('saving the model during epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save(f'i{total_steps}')
                    self.save(f'latest')


                # Save camera_pose_dict during optimization
                if opts.save_partial_camera_pose_dict:
                    if (len(self.datasetCameraPoseDict)!=len(self.dataloader.dataset)):
                        print(f'Saving cam_pose_dict of size {len(self.datasetCameraPoseDict)}!={len(self.dataloader.dataset)}.')
                    tb_visualizer.save_raw_stats({'campose':self.datasetCameraPoseDict}, 'campose_partial', total_steps)

                if total_steps == opts.num_iter:
                    return

            epoch_stats = self.get_epoch_statistics()
            tb_visualizer.display_current_results(epoch_stats, epoch)
            tb_visualizer.save_raw_stats(epoch_stats['raw'], 'raw', epoch)

            # Save camera_pose_dict after optimizing
            if opts.save_camera_pose_dict:
                if (len(self.datasetCameraPoseDict)!=len(self.dataloader.dataset)):
                    print('Saving cam_pose_dict of size ',len(self.datasetCameraPoseDict),f'!={len(self.dataloader.dataset)}.')
                tb_visualizer.save_raw_stats({'campose':self.datasetCameraPoseDict}, 'campose', epoch)

            if self.train_mode and ((epoch+1) % opts.save_epoch_freq == 0):
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)


        self.save('latest')
        self.save(opts.num_epochs)

