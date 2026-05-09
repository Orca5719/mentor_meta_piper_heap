import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import shutil

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import metaworld_env as mw

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re

from utils import models_tuple
from copy import deepcopy

torch.backends.cudnn.benchmark = True


def _make_env(cfg):
    """Create environment based on real_mode config.

    If real_mode is enabled, use PiPER real robot environment;
    otherwise, use MetaWorld simulation environment.
    """
    if getattr(cfg, 'real_mode', False):
        import piper_env as pe
        real_cfg = getattr(cfg, 'real', None)
        kwargs = {}
        if real_cfg is not None:
            kwargs = {
                'can_port': getattr(real_cfg, 'can_port', 'can0'),
                'action_scale': getattr(real_cfg, 'action_scale', 2.0),
                'gripper_range': getattr(real_cfg, 'gripper_range', 0.08),
                'speed': getattr(real_cfg, 'speed', 50),
                'max_episode_steps': getattr(cfg, 'episode_length', 250),
                'use_realsense': getattr(real_cfg, 'use_realsense', True),
                'realsense_serial': getattr(real_cfg, 'realsense_serial', None),
                'realsense_resolution': list(getattr(real_cfg, 'realsense_resolution', [640, 480])),
                'realsense_fps': getattr(real_cfg, 'realsense_fps', 30),
                'calibration_file': getattr(real_cfg, 'calibration_file', None),
                'manual_reward': getattr(real_cfg, 'manual_reward', True),
            }
            # Workspace bounds (mm)
            if hasattr(real_cfg, 'workspace'):
                ws = real_cfg.workspace
                kwargs['workspace'] = {
                    'x_min': getattr(ws, 'x_min', -200),
                    'x_max': getattr(ws, 'x_max', 200),
                    'y_min': getattr(ws, 'y_min', -200),
                    'y_max': getattr(ws, 'y_max', 200),
                    'z_min': getattr(ws, 'z_min', 50),
                    'z_max': getattr(ws, 'z_max', 400),
                }
            # Home pose (mm, deg)
            if hasattr(real_cfg, 'home_pose'):
                kwargs['home_pose'] = np.array(real_cfg.home_pose)
        return pe.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed, **kwargs)
    else:
        return mw.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.seed)


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        print("#"*20)
        print(f'\nworkspace: {self.work_dir}')
        print(self.cfg)
        self.last_save_step = -9999
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            name_1 = cfg.task_name
            name_2 = group_name
            try:
                name_2 += '_' + cfg.title
            except:
                pass
            name_3 = exp_name
            wandb.init(project=name_1,
                       group=name_2,
                       name=name_3,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = cfg.discount_alpha
        self._discount_alpha_temp = cfg.discount_alpha_temp
        self._discount_beta = cfg.discount_beta
        self._discount_beta_temp = cfg.discount_beta_temp
        self._nstep = cfg.nstep
        self._nstep_alpha = cfg.nstep_alpha
        self._nstep_alpha_temp = cfg.nstep_alpha_temp
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = _make_env(self.cfg)
        if getattr(self.cfg, 'real_mode', False):
            # Real mode: share the same env (single robot), no separate eval env
            self.eval_env = self.train_env
            print("[Real Mode] Using PiPER robot for both train and eval")
        else:
            self.eval_env = _make_env(self.cfg)
        # create replay buffer (with is_intervened field)
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'),
                      specs.Array((1, ), np.float32, 'is_intervened'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        # load pretrained buffer if specified
        if self.cfg.pretrain_buffer_dir is not None:
            self._load_pretrain_buffer(self.cfg.pretrain_buffer_dir)
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            math.floor(self._nstep + self._nstep_alpha),
            self._discount - self._discount_alpha - self._discount_beta)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step /
            self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def _load_pretrain_buffer(self, pretrain_dir):
        """Copy pretrained buffer .npz files into current replay buffer directory."""
        pretrain_path = Path(pretrain_dir)
        if not pretrain_path.exists():
            print(f'[WARNING] Pretrain buffer dir not found: {pretrain_path}')
            return
        buffer_dir = self.work_dir / 'buffer'
        npz_files = list(pretrain_path.glob('*.npz'))
        if len(npz_files) == 0:
            print(f'[WARNING] No .npz files found in {pretrain_path}')
            return
        for src in npz_files:
            dst = buffer_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
        print(f'[INFO] Loaded {len(npz_files)} pretrained buffer episodes from {pretrain_path}')

    def pretrain_bc(self):
        """Behavior Cloning pretraining on demo data.

        Runs BC updates for cfg.bc_pretrain_steps using the preloaded
        demo buffer. Only encoder + actor are updated; critic and
        value_predictor remain at random init (they will be trained
        during the subsequent RL phase).

        Must be called after setup() and agent creation, and before train().
        """
        bc_steps = getattr(self.cfg, 'bc_pretrain_steps', 0)
        bc_loss_type = getattr(self.cfg, 'bc_loss_type', 'mse')
        bc_log_interval = getattr(self.cfg, 'bc_log_interval', 500)

        if bc_steps <= 0:
            return

        # Check that buffer has demo data
        buffer_size = len(self.replay_storage)
        if buffer_size == 0:
            print('[BC] WARNING: Replay buffer is empty, skipping BC pretraining')
            return

        print(f"\n{'='*60}")
        print(f"  BC Pretraining: {bc_steps} steps, loss_type={bc_loss_type}")
        print(f"  Demo buffer size: {buffer_size} transitions")
        print(f"{'='*60}")

        # Need to fetch buffer data into replay loader
        self._replay_iter = iter(self.replay_loader)

        for bc_step in range(1, bc_steps + 1):
            metrics = self.agent.update_bc(self._replay_iter, bc_loss_type)

            if bc_step % bc_log_interval == 0 or bc_step == 1:
                print(f"[BC] step {bc_step}/{bc_steps} | "
                      f"bc_loss={metrics.get('bc_loss', 0):.4f} | "
                      f"action_error={metrics.get('bc_action_error', 0):.4f}")
                if self.cfg.use_tb or self.cfg.use_wandb:
                    with self.logger.log_and_dump_ctx(bc_step, ty='bc') as log:
                        log('bc_loss', metrics.get('bc_loss', 0))
                        log('bc_action_error', metrics.get('bc_action_error', 0))
                        log('bc_step', bc_step)

        # Save BC pretrained snapshot
        self.save_snapshot(step_id='bc_pretrained')
        print(f"[BC] Pretraining done. Snapshot saved as snapshot_bc_pretrained.pt\n")

    def update_buffer(self):
        #self.buffer.update_discount(self.discount)
        self.buffer.update_nstep(self.nstep)
        return
    
    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_sr = False
            time_step = self.eval_env.reset()
            # self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            self.video_recorder.init(self.eval_env, enabled=False)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                episode_sr = episode_sr or time_step.success
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            total_sr += episode_sr
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def _init_spacemouse(self):
        """Initialize Spacemouse reader for real-mode intervention."""
        spacemouse_cfg = getattr(self.cfg, 'spacemouse', None)
        if spacemouse_cfg is None or not getattr(spacemouse_cfg, 'enabled', False):
            return None
        from spacemouse_reader import SpacemouseReader
        reader = SpacemouseReader(
            dead_zone=getattr(spacemouse_cfg, 'dead_zone', 0.1),
        )
        reader.start()
        return reader

    def train(self):
        # predicates
        # frames = steps * action_repeat
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # Initialize Spacemouse for real-mode intervention
        spacemouse_reader = None
        if getattr(self.cfg, 'real_mode', False):
            spacemouse_reader = self._init_spacemouse()

        episode_step, episode_reward, episode_sr = 0, 0, False
        episode_interventions = 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        print("start training")
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                # update priority queue
                if hasattr(self.agent, 'tp_set'):
                    self.agent.tp_set.add(episode_reward,\
                                            deepcopy(self.agent.actor),\
                                            deepcopy(self.agent.critic),\
                                            deepcopy(self.agent.critic_target),\
                                            deepcopy(self.agent.value_predictor),\
                                            moe=deepcopy(self.agent.actor.moe.experts),\
                                            gate=deepcopy(self.agent.actor.moe.gate))                    
                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                if self.cfg.save_snapshot and self.global_step - self.last_save_step >= self.cfg.save_interval:
                    self.last_save_step = self.global_step
                    self.save_snapshot(self.global_step)
                episode_sr = False
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # Spacemouse intervention check (available throughout training)
            is_intervened = False
            if spacemouse_reader is not None:
                sm_action, is_intervening = spacemouse_reader.get_action()
                if is_intervening and sm_action is not None:
                    # Map Spacemouse 7D to PiperEnv 4D: [dx, dy, dz, gripper]
                    action = sm_action[[0, 1, 2, 6]]
                    is_intervened = True

            # try to update the agent
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict()
                if hasattr(self.agent, 'tp_set'):
                    metrics = self.agent.tp_set.log(metrics)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            # Attach intervention flag to time_step before storing
            if is_intervened:
                time_step = time_step._replace(is_intervened=np.array([1.0], dtype=np.float32))
                episode_interventions += 1
            else:
                time_step = time_step._replace(is_intervened=np.array([0.0], dtype=np.float32))
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, step_id=None):
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            if not os.path.exists(str(self.work_dir) + '/snapshots'):
                os.makedirs(str(self.work_dir) + '/snapshots')
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step_id=None):
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
        if not snapshot.exists():
            raise FileNotFoundError(f"Snapshot {snapshot} not found.")
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs):
    from train_mw import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfgs)
    if cfgs.load_from_id:
        snapshot = root_dir / 'snapshots' / f'snapshot_{cfgs.load_id}.pt'
    else:
        snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    # BC pretraining on demo data (before RL)
    workspace.pretrain_bc()
    workspace.train()


if __name__ == '__main__':
    main()
