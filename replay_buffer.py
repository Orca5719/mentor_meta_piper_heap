import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def _episode_demo_ratio(episode):
    """Compute intervention ratio for an episode (0.0 = pure RL, 1.0 = pure demo)."""
    is_int = episode.get('is_intervened', None)
    if is_int is None:
        return 0.0
    # skip the dummy first transition (index 0)
    return float(is_int[1:].mean())


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


def _episode_is_negative(episode):
    """Check if an episode is a negative sample (no staged reward triggered)."""
    rewards = episode.get('reward', None)
    if rewards is None:
        return True
    # skip the dummy first transition (index 0)
    return bool((rewards[1:] < 0.5).all())


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot,
                 demo_ratio_start=1.0, demo_ratio_end=0.3, demo_ratio_decay_steps=50000,
                 negative_ratio=0.2):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._episode_demo_ratios = dict()  # eps_fn -> demo_ratio
        self._episode_neg_flags = dict()    # eps_fn -> is_negative
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

        # Demo优先采样参数
        self._demo_ratio_start = demo_ratio_start
        self._demo_ratio_end = demo_ratio_end
        self._demo_ratio_decay_steps = demo_ratio_decay_steps
        self._total_samples = 0

        # 负样本采样比例（占RL采样量的比例）
        self._negative_ratio = negative_ratio

    def _compute_demo_ratio(self):
        """Compute current demo sampling ratio based on decay schedule."""
        if self._demo_ratio_decay_steps <= 0:
            return self._demo_ratio_end
        t = min(1.0, self._total_samples / self._demo_ratio_decay_steps)
        return self._demo_ratio_start + (self._demo_ratio_end - self._demo_ratio_start) * t

    def _is_demo_episode(self, eps_fn):
        """Check if an episode is a demo episode (intervention ratio > 0.3)."""
        ratio = self._episode_demo_ratios.get(eps_fn, 0.0)
        return ratio > 0.3

    def _is_negative_episode(self, eps_fn):
        """Check if an episode is a negative sample (no staged reward triggered)."""
        return self._episode_neg_flags.get(eps_fn, False)

    def _sample_episode(self):
        if len(self._episode_fns) == 0:
            return None

        current_demo_ratio = self._compute_demo_ratio()

        # Separate into three pools: demo, negative_rl, positive_rl
        demo_fns = []
        negative_fns = []
        positive_fns = []
        for fn in self._episode_fns:
            if self._is_demo_episode(fn):
                demo_fns.append(fn)
            elif self._is_negative_episode(fn):
                negative_fns.append(fn)
            else:
                positive_fns.append(fn)

        # Decide which pool to sample from
        roll = random.random()

        if roll < current_demo_ratio and demo_fns:
            # Sample from demo episodes, weighted by intervention ratio
            weights = np.array([self._episode_demo_ratios[fn] for fn in demo_fns])
            weights = weights / weights.sum()
            idx = np.random.choice(len(demo_fns), p=weights)
            return self._episodes[demo_fns[idx]]

        # RL portion: split between negative and positive
        rl_roll = random.random()
        if rl_roll < self._negative_ratio and negative_fns:
            eps_fn = random.choice(negative_fns)
            return self._episodes[eps_fn]

        if positive_fns:
            eps_fn = random.choice(positive_fns)
            return self._episodes[eps_fn]

        # Fallback: sample from whatever is available
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)

        # Track demo ratio and negative flag for this episode
        self._episode_demo_ratios[eps_fn] = _episode_demo_ratio(episode)
        self._episode_neg_flags[eps_fn] = _episode_is_negative(episode)

        while eps_len + self._size > self._max_size:
            # Find the best candidate to evict: prefer non-demo episodes, then oldest
            evict_idx = self._find_evict_candidate()
            if evict_idx is None:
                break
            evict_fn = self._episode_fns.pop(evict_idx)
            evict_eps = self._episodes.pop(evict_fn)
            self._episode_demo_ratios.pop(evict_fn, None)
            self._episode_neg_flags.pop(evict_fn, None)
            self._size -= episode_len(evict_eps)
            evict_fn.unlink(missing_ok=True)

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _find_evict_candidate(self):
        """Find the best episode to evict: prefer negative > non-demo > oldest."""
        # First try to find a negative (non-demo) episode
        for i, fn in enumerate(self._episode_fns):
            if not self._is_demo_episode(fn) and self._is_negative_episode(fn):
                return i
        # Then try any non-demo episode
        for i, fn in enumerate(self._episode_fns):
            if not self._is_demo_episode(fn):
                return i
        # All are demo episodes — evict the oldest (index 0)
        return 0 if self._episode_fns else None

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        self._total_samples += 1
        episode = self._sample_episode()
        if episode is None:
            return None
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        # is_intervened flag (default 0.0 for episodes without this field)
        is_intervened = episode.get('is_intervened', np.zeros_like(episode['reward']))[idx]
        return (obs, action, reward, discount, next_obs, is_intervened)

    def __iter__(self):
        while True:
            yield self._sample()

    def update_nstep(self, new_nstep):
        self._nstep = new_nstep

    def update_discount(self, new_discount):
        self._discount = new_discount


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount,
                       demo_ratio_start=1.0, demo_ratio_end=0.3, demo_ratio_decay_steps=50000,
                       negative_ratio=0.2):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            demo_ratio_start=demo_ratio_start,
                            demo_ratio_end=demo_ratio_end,
                            demo_ratio_decay_steps=demo_ratio_decay_steps,
                            negative_ratio=negative_ratio)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader, iterable
