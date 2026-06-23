import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import time
import gc
import cv2
import json
import shutil
import numpy as np
import torch
import threading
import concurrent.futures
from collections import deque, namedtuple
from pathlib import Path
from dm_env import StepType, specs
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage, make_replay_loader
    import utils
    from logger import Logger
    from piper_env import PiperRobot
    from realsense_camera import RealSenseCamera
    from spacemouse_reader import SpacemouseReader
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)

import base64
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

_TimeStepBase = namedtuple('_TimeStepBase', [
    'observation', 'action', 'reward', 'discount', 'first', 'is_last', 'is_intervened'
])

class TimeStep(_TimeStepBase):
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)

    def last(self):
        return self.is_last


class DoubaoGoalRewarder:
    """基于豆包 VLM 的分段自动目标检测奖励器。

    支持三阶段判定（reached/grasped/lifted），每个阶段对应一张目标图片。
    VLM 只判断当前尚未达到的下一阶段，匹配则推进阶段并触发对应分段奖励。

    目标图片默认存放在 images/ 目录下：
        images/target_reached.jpg  — 阶段1: 机械臂到达物体附近
        images/target_grasped.jpg  — 阶段2: 机械臂抓住物体
        images/target_lifted.jpg   — 阶段3: 机械臂提起物体
    """

    # ========== 在这里填写豆包 API Key ==========
    DOUBAO_API_KEY = 'API'
    # ============================================

    DOUBAO_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
    DOUBAO_MODEL = 'ep-20260606175321-9sh48'

    STAGE_NAMES = {1: 'reached', 2: 'grasped', 3: 'lifted'}
    STAGE_PROMPTS = {
        1: '你正在观察一个机械臂操作的实时画面。'
           '请判断当前画面中，机械臂夹爪是否已经接近桌面上的物体（红色方块）。'
           '判断标准：夹爪末端与物体之间的距离是否已经很近（夹爪在物体正上方或旁边），不需要抓住。'
           '注意：物体在桌面上的位置不固定，只需关注夹爪与物体的相对距离。'
           '如果夹爪已接近物体，is_match=true，否则为 false。'
           '必须只输出 JSON: {"is_match": true/false, "confidence": 0.0-1.0, "reason": "简短中文原因"}',
        2: '你正在观察一个机械臂操作的实时画面。'
           '请判断当前画面中，机械臂夹爪是否已经夹住物体（红色方块）。'
           '关键判断标准：夹爪的两个指是否已经闭合，且物体（红色方块）被夹在两个指之间。'
           '与"仅接近"的区别："接近"时夹爪是张开的（两指之间有明显间距），"夹住"时夹爪两指已合拢，物体被卡在中间。'
           '请仔细观察夹爪两指之间的间距：如果间距大、物体在夹爪外面，说明只是接近未夹住；如果间距小、物体在两指之间，说明已夹住。'
           '如果夹爪已闭合且物体在两指之间，is_match=true，否则为 false。'
           '必须只输出 JSON: {"is_match": true/false, "confidence": 0.0-1.0, "reason": "简短中文原因"}',
        3: '你正在观察一个机械臂操作的实时画面。'
           '请判断当前画面中，机械臂是否已经成功提起物体（红色方块）离开桌面。'
           '判断标准：物体是否已经离开桌面，被夹爪悬空持住。'
           '注意：物体被提起后的位置不固定，只需关注物体是否离开桌面。'
           '如果物体已离开桌面并被夹持悬空，is_match=true，否则为 false。'
           '必须只输出 JSON: {"is_match": true/false, "confidence": 0.0-1.0, "reason": "简短中文原因"}',
    }

    def __init__(self, model=None, images_dir='images', output_dir='.',
                 confidence_threshold=0.60, check_every_steps=5,
                 api_key=None, base_url=None):
        self.model = model or self.DOUBAO_MODEL
        _PLACEHOLDER = '在这里填写你的豆包API Key'
        self.api_key = api_key if api_key and api_key != _PLACEHOLDER else self.DOUBAO_API_KEY
        self.base_url = base_url or self.DOUBAO_BASE_URL
        if not self.api_key or self.api_key == _PLACEHOLDER:
            raise ValueError(
                '请先在 DoubaoGoalRewarder.DOUBAO_API_KEY 中填写你的豆包 API Key，'
                '或通过 --doubao_api_key 参数传入'
            )
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = float(confidence_threshold)
        self.check_every_steps = max(1, int(check_every_steps))
        self.current_frame_path = self.output_dir / 'vlm_current_frame.jpg'
        self.target_paths = self._prepare_target_images()
        self.last_result = self._default_result(stage=0, reason='尚未开始检测')
        self._vlm_future = None

        # 初始化 OpenAI 客户端（豆包兼容 OpenAI API）
        if OpenAI is not None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = None

    def _default_result(self, stage=0, reason='未检测'):
        return {
            'stage': stage,
            'is_match': False,
            'confidence': 0.0,
            'reason': reason,
            'raw': ''
        }

    def reset(self):
        """重置检测状态，在每个新 episode 开始时调用。"""
        # 等待异步请求完成（最多等 2 秒），避免旧结果污染新 episode
        if hasattr(self, '_vlm_future') and self._vlm_future is not None:
            try:
                self._vlm_future.result(timeout=2.0)
            except Exception:
                pass
            self._vlm_future = None
        self.last_result = self._default_result(stage=0, reason='新 episode，尚未检测')

    def _find_image(self, stage):
        """在 images_dir 下查找指定阶段的目标图片（支持 jpg/png/jpeg/bmp/webp）。"""
        name = f'target_{self.STAGE_NAMES[stage]}'
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            candidate = self.images_dir / f'{name}{ext}'
            if candidate.exists():
                return candidate
        return None

    def _prepare_target_images(self):
        """加载并复制三阶段目标图片到输出目录。"""
        paths = {}
        missing = []
        for stage in [1, 2, 3]:
            src = self._find_image(stage)
            if src is None:
                missing.append(f'target_{self.STAGE_NAMES[stage]}.jpg')
                continue
            dst = self.output_dir / f'target_{self.STAGE_NAMES[stage]}{src.suffix}'
            if src.resolve() != dst.resolve():
                shutil.copy2(str(src), str(dst))
            paths[stage] = str(dst)

        if missing:
            raise FileNotFoundError(
                f'缺少目标图片，请将以下图片放入 {self.images_dir.resolve()}/ 目录: '
                f'{", ".join(missing)}'
            )
        return paths

    def _extract_json(self, content):
        content = (content or '').strip()
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end + 1])
            raise

    def _coerce_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'y')
        if isinstance(value, (int, float)):
            return value != 0
        return False

    def _coerce_confidence(self, value):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1.0:
            confidence = confidence / 100.0
        return max(0.0, min(1.0, confidence))

    def _encode_image_base64(self, image_path):
        """将图片文件编码为 base64 data URL。"""
        ext = Path(image_path).suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.png': 'image/png', '.bmp': 'image/bmp', '.webp': 'image/webp'}
        mime = mime_map.get(ext, 'image/jpeg')
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f'data:{mime};base64,{data}'

    def evaluate(self, frame_rgb, current_stage, episode, step, camera=None,
                 gripper_open=True, arm_z=None, home_z=None):
        """异步评估当前帧是否达到下一个阶段（非阻塞）。

        VLM 调用在后台线程执行，主循环不会被阻塞。
        返回上一次完成的检测结果，当前请求异步提交。

        Args:
            frame_rgb: 当前RGB帧（84x84，仅供参考，VLM 使用高分辨率帧）
            current_stage: 当前已达到的阶段 (0=none, 1=reached, 2=grasped)
            episode: 当前episode编号
            step: 当前全局步数
            camera: RealSenseCamera 实例，用于捕获高分辨率帧
            gripper_open: 夹爪是否张开（True=张开）
            arm_z: 当前机械臂Z坐标
            home_z: 复位时Z坐标

        Returns:
            dict: {'stage', 'is_match', 'confidence', 'reason', 'raw'}
                  stage=匹配的阶段编号(1/2/3)，未匹配时等于current_stage
        """
        if current_stage >= 3:
            return self.last_result

        if OpenAI is None or self._client is None:
            result = self._default_result(stage=current_stage, reason='未安装 openai Python 包')
            self.last_result = result
            return result

        # 每步都检查异步结果是否完成，完成就立即收取（减少延迟）
        if hasattr(self, '_vlm_future') and self._vlm_future is not None and self._vlm_future.done():
            try:
                result = self._vlm_future.result()
                if result.get('stage', 0) > current_stage + 1:
                    result = self._default_result(stage=current_stage, reason='旧结果已过期')
                self.last_result = result
            except Exception as exc:
                self.last_result = self._default_result(stage=current_stage, reason=f'VLM 检测失败: {exc}')
            self._vlm_future = None

        # 只在 check_every_steps 间隔提交新请求
        if step % self.check_every_steps != 0:
            return self.last_result

        # 如果上一次异步调用还没完成，跳过本次提交
        if self._vlm_future is not None and not self._vlm_future.done():
            return self.last_result

        # 提交新的异步请求：使用高分辨率帧（如果 camera 可用）
        next_stage = current_stage + 1

        # 前置过滤：根据机械臂状态跳过不可能的阶段
        if next_stage == 2 and gripper_open:
            # 夹爪张开时不可能夹住物体，跳过VLM调用
            return self.last_result
        if next_stage == 3 and home_z is not None and arm_z is not None:
            # Z轴没有抬高时不可能提起物体（阈值10mm）
            if arm_z < home_z - 50:
                return self.last_result

        try:
            if camera is not None and hasattr(camera, 'capture_full'):
                hi_res_frame = camera.capture_full()  # 640x480 RGB
                frame_bgr = cv2.cvtColor(hi_res_frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.current_frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        except Exception:
            return self.last_result

        self._vlm_future = self._submit_async(current_stage, next_stage)
        return self.last_result

    def _submit_async(self, current_stage, next_stage):
        """在线程池中异步提交 VLM 请求。"""
        if not hasattr(self, '_vlm_executor'):
            self._vlm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self._vlm_executor.submit(self._call_vlm, current_stage, next_stage)

    def _call_vlm(self, current_stage, next_stage):
        """实际执行 VLM API 调用（在后台线程中运行）。"""
        try:
            prompt = self.STAGE_PROMPTS[next_stage]
            current_b64 = self._encode_image_base64(str(self.current_frame_path))

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': current_b64}},
                        {'type': 'text', 'text': prompt},
                    ]
                }],
                temperature=0,
            )

            raw_content = response.choices[0].message.content

            parsed = self._extract_json(raw_content)
            if not parsed:
                print(f"[VLM] JSON解析失败 | raw={raw_content[:200]}")
                return self._default_result(stage=current_stage, reason='JSON解析失败')
            is_match = self._coerce_bool(
                parsed.get('is_match', parsed.get('matched', False))
            )
            confidence = self._coerce_confidence(parsed.get('confidence', 0.0))
            reason = str(parsed.get('reason', ''))

            matched_stage = next_stage if (is_match and confidence >= self.confidence_threshold) else current_stage

            # VLM 调用结果日志
            if is_match:
                print(f"[VLM] stage {current_stage}→{next_stage} | match={is_match} "
                      f"conf={confidence:.2f} → {matched_stage} | reason={reason[:80]}")
            elif confidence > 0.3:
                print(f"[VLM] stage {current_stage}→{next_stage} | match={is_match} "
                      f"conf={confidence:.2f} (低置信度) | reason={reason[:80]}")

            return {
                'stage': matched_stage,
                'is_match': is_match,
                'confidence': confidence,
                'reason': reason[:120],
                'raw': raw_content
            }
        except Exception as exc:
            print(f"[VLM] FAILED stage {current_stage}→{next_stage} | error={exc}")
            return self._default_result(stage=current_stage, reason=f'VLM 检测失败: {exc}')


class PiperRobotTrainer:
    def __init__(self):
        print("="*60)
        print("     Piper 机械臂实时训练 (兼容 mentor piper_env)")
        print("="*60)

        self.work_dir = Path.cwd()

        self.IMG_HEIGHT = 84
        self.IMG_WIDTH = 84
        self.frame_stack = 3
        self.batch_size = 256
        self.update_every_episodes = 5
        self.save_interval = 1000
        self.seed_steps = 1000

        self.camera = None
        self.piper_arm = None
        self.agent = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._global_step = 0
        self._global_episode = 0
        self.last_save_step = -9999

        self.frames_queue = deque(maxlen=3)
        self.replay_storage = None
        self.replay_loader = None
        self.replay_iter = None

        self._reward = 0.0
        self.episode_reward = 0.0
        self.episode_step = 0
        self._last_action = None
        self._obs_spec = None
        self._act_spec = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = True

        # 机械臂初始位姿 (mm, deg 浮点单位，与 piper_env.py 一致)
        self.HOME_X, self.HOME_Y, self.HOME_Z = 300.614, -12.185, 282.341
        self.HOME_RX, self.HOME_RY, self.HOME_RZ = -179.351, 23.933, 177.934
        self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
        self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
        self.gripper_open = True

        self._buffer_dir = Path.cwd() / 'buffer_robot'

        # 动作参数 (mm 浮点单位)
        self.action_scale = 2.0   # mm per action unit，与 piper_env.py 的 action_scale 一致
        self.action_interval = 0.08  # 最小动作间隔(秒)

        # 安全范围 (mm)
        self.WS_X_MIN, self.WS_X_MAX = 150.0, 450.0
        self.WS_Y_MIN, self.WS_Y_MAX = -150.0, 150.0
        self.WS_Z_MIN, self.WS_Z_MAX = 150.0, 350.0

        # 时间惩罚：非干预帧每秒-0.33 reward，与631总额10匹配
        self.time_penalty_per_step = -0.33 * self.action_interval  # ≈ -0.026/步

        self.random_amplitude = 0.6
        self.random_drift_prob = 0.2
        self.last_random_direction = np.zeros(3)

        # 输出目录
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path.cwd() / "piper_outputs" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 训练输出目录: {self.output_dir.resolve()}")

        # 3D鼠标
        self.DEAD_ZONE = 0.15
        self.SPACE_MOUSE_ACTION_SCALE = 2.0
        self.spacemouse_reader = None
        self.is_intervening = False

        # Staged reward
        self.STAGE_REWARDS = {1: 6.0, 2: 3.0, 3: 1.0}
        self.STAGE_NAMES = {0: "none", 1: "reached", 2: "grasped", 3: "lifted"}
        self.current_stage = 0

        # 夹爪
        self.random_gripper_state = 1.0
        self.last_gripper_change_step = 0
        self.gripper_change_interval = 100

        self.demo_ratio_start = 1.0
        self.demo_ratio_end = 0.3
        self.demo_ratio_decay_steps = 50000
        self.negative_ratio = 0.2

        # 防卡死机制
        self.last_action_time = time.time()
        self.arm_command_lock = threading.Lock()
        self.heartbeat_timeout = 2.0
        self.last_heartbeat = time.time()

        # 闭环位置反馈
        self.use_closed_loop = True

        # 异步训练参数
        self.async_update = True
        self.async_update_interval = 0.01  # 异步更新间隔(秒)
        self._async_update_count = 0
        self._async_train_thread = None
        self._async_pause = threading.Event()  # 请求异步线程暂停
        self._async_paused = threading.Event()  # 异步线程确认已暂停
        self._async_pause.clear()
        self._async_paused.clear()

        # VLM 自动奖励
        self.use_vlm_reward = False
        self.doubao_api_key = ''
        self.doubao_model = 'API'
        self.doubao_base_url = 'https://ark.cn-beijing.volces.com/api/v3'
        self.images_dir = str(Path(script_dir) / 'images')
        self.vlm_confidence_threshold = 0.60
        self.vlm_check_every = 20
        self.vlm_rewarder = None
        self.last_vlm_result = None

    def _init_specs(self):
        """直接硬编码spec，与 piper_env.py 的 piper_wrapper 输出一致。"""
        self._obs_spec = specs.BoundedArray(
            shape=(9, self.IMG_HEIGHT, self.IMG_WIDTH),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        self._act_spec = specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )

    def _init_replay_storage(self):
        """初始化ReplayBuffer，spec与 train_mw.py 一致（5个字段，含is_intervened）。"""
        self._init_specs()

        data_specs = (
            specs.Array(self._obs_spec.shape, self._obs_spec.dtype, 'observation'),
            specs.Array(self._act_spec.shape, self._act_spec.dtype, 'action'),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
            specs.Array((1,), np.float32, 'is_intervened'),
        )

        self._buffer_dir.mkdir(exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)

        self.replay_loader, self._replay_buffer = make_replay_loader(
            self._buffer_dir, max_size=100000, batch_size=self.batch_size, num_workers=0,
            save_snapshot=True, nstep=3, discount=0.99,
            demo_ratio_start=self.demo_ratio_start,
            demo_ratio_end=self.demo_ratio_end,
            demo_ratio_decay_steps=self.demo_ratio_decay_steps,
            negative_ratio=self.negative_ratio,
        )
        self.replay_iter = iter(self.replay_loader)

        print("✅ 回放缓冲区初始化完成")

    def init_hardware(self):
        """初始化硬件，全部使用本地 piper_env.py / realsense_camera.py 接口。"""
        print("初始化硬件...")

        # 相机：使用 RealSenseCamera
        self.camera = RealSenseCamera(
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH)
        )
        self.camera.connect()

        # 机械臂：使用 piper_env.PiperRobot
        self.piper_arm = PiperRobot("can0")
        self.piper_arm.connect()
        self.piper_arm.enable()

        # 清除所有错误状态
        time.sleep(0.5)

        # 移动到初始位姿
        self.piper_arm.move_to_pose(
            self.HOME_X, self.HOME_Y, self.HOME_Z,
            self.HOME_RX, self.HOME_RY, self.HOME_RZ,
            speed=50
        )
        self.piper_arm.set_gripper(0.08)  # 打开夹爪
        time.sleep(2.0)

        # 读取实际位置，同步软件位置
        self._sync_actual_position()

        self._init_replay_storage()

        # VLM 自动奖励初始化
        if self.use_vlm_reward:
            try:
                self.vlm_rewarder = DoubaoGoalRewarder(
                    model=self.doubao_model,
                    images_dir=self.images_dir,
                    output_dir=self.output_dir,
                    confidence_threshold=self.vlm_confidence_threshold,
                    check_every_steps=self.vlm_check_every,
                    api_key=self.doubao_api_key or None,
                    base_url=self.doubao_base_url or None,
                )
                self.last_vlm_result = self.vlm_rewarder.last_result
                print(f"✅ VLM 自动奖励已启用 | model={self.doubao_model}")
                print(f"✅ 目标图片目录: {self.vlm_rewarder.images_dir.resolve()}")
                for stage, path in self.vlm_rewarder.target_paths.items():
                    print(f"   阶段{stage}({DoubaoGoalRewarder.STAGE_NAMES[stage]}): {Path(path).name}")
            except Exception as e:
                print(f"⚠️  VLM 自动奖励初始化失败: {e}")
                self.use_vlm_reward = False
                self.vlm_rewarder = None

        # 3D鼠标：使用 SpacemouseReader（非阻塞）
        self.spacemouse_reader = SpacemouseReader(
            dead_zone=self.DEAD_ZONE,
            action_scale=self.SPACE_MOUSE_ACTION_SCALE
        )
        self.spacemouse_reader.start()
        time.sleep(1.0)

        # 图像线程
        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

        # 机械臂状态监控线程
        threading.Thread(target=self._arm_status_monitor_thread, daemon=True).start()

        # 异步训练线程
        self._start_async_train()

        print("✅ 硬件初始化完成")
        print()

    def _sync_actual_position(self):
        """读取机械臂实际位置，同步到软件变量（mm 浮点单位）"""
        with self.arm_command_lock:
            try:
                actual_pose = self.piper_arm.get_end_pose()  # 返回 [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]
                if actual_pose is not None and len(actual_pose) >= 6:
                    self.X = actual_pose[0]
                    self.Y = actual_pose[1]
                    self.Z = actual_pose[2]
                    self.RX = actual_pose[3]
                    self.RY = actual_pose[4]
                    self.RZ = actual_pose[5]
                    print(f"✅ 位置同步成功: X={self.X:.1f}, Y={self.Y:.1f}, Z={self.Z:.1f}")
            except Exception as e:
                print(f"⚠️  位置同步失败: {e}")

    def _arm_status_monitor_thread(self):
        """后台监控机械臂状态，自动复位保护"""
        while self._running:
            try:
                with self.arm_command_lock:
                    if not self.piper_arm.check_safety():
                        print("\n❌ 机械臂触发保护！")
                        print("正在自动复位...")

                        # PiperRobot.check_safety() 已经在严重情况下调用了 emergency_stop
                        # 需要重新使能
                        try:
                            self.piper_arm.enable()
                        except:
                            pass
                        time.sleep(0.2)

                        # 同步实际位置
                        self._sync_actual_position()

                        print("✅ 机械臂自动复位完成")

                time.sleep(0.5)
            except Exception as e:
                time.sleep(1.0)

    def _image_thread(self):
        while self._running:
            try:
                frame = self.camera.capture()  # RealSenseCamera.capture() 已 resize，返回 RGB
                if frame is not None:
                    with self._lock:
                        self.frames_queue.append(frame)
                        self._latest_frame = frame.copy()
            except Exception as e:
                print(f"[图像线程错误] {e}")
            time.sleep(0.005)

    def _train_thread(self):
        """后台异步梯度更新线程"""
        print("[异步训练] 线程已启动，等待seed阶段完成...")
        while self._running:
            if self.agent is None or self._global_step < self.seed_steps:
                time.sleep(0.5)
                continue
            if self.replay_iter is None:
                time.sleep(0.5)
                continue
            # 被请求暂停时，确认暂停并等待恢复
            if self._async_pause.is_set():
                self._async_paused.set()  # 通知主线程：我已停下
                time.sleep(0.01)
                continue
            self._async_paused.clear()
            try:
                self.agent.update(self.replay_iter, self._global_step)
                self._async_update_count += 1
            except Exception as e:
                pass  # 采样失败静默重试
            time.sleep(self.async_update_interval)

    def _start_async_train(self):
        """启动异步训练线程"""
        if not self.async_update:
            return
        self._async_train_thread = threading.Thread(target=self._train_thread, daemon=True)
        self._async_train_thread.start()
        print("✅ 异步训练线程已启动")

    def get_stacked_obs(self):
        with self._lock:
            frames = list(self.frames_queue)

        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))

        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))

        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    def apply_action(self, action):
        """
        安全版动作执行（mm 浮点单位）：
        1. 闭环位置控制
        2. 速度限制
        3. 工作空间裁剪
        """
        now = time.time()
        if now - self.last_action_time < self.action_interval:
            return
        self.last_action_time = now
        self.last_heartbeat = now

        with self.arm_command_lock:
            try:
                # 先同步实际位置（闭环控制）
                if self.use_closed_loop:
                    actual_pose = self.piper_arm.get_end_pose()
                    if actual_pose is not None and len(actual_pose) >= 6:
                        self.X = actual_pose[0]
                        self.Y = actual_pose[1]
                        self.Z = actual_pose[2]

                # 计算目标位置 (action ∈ [-1,1] * action_scale mm)
                dx = action[0] * self.action_scale
                dy = action[1] * self.action_scale
                dz = action[2] * self.action_scale

                # 工作空间裁剪
                target_X = np.clip(self.X + dx, self.WS_X_MIN, self.WS_X_MAX)
                target_Y = np.clip(self.Y + dy, self.WS_Y_MIN, self.WS_Y_MAX)
                target_Z = np.clip(self.Z + dz, self.WS_Z_MIN, self.WS_Z_MAX)

                # 夹爪控制
                if action[3] > 0:
                    if not self.gripper_open:
                        self.gripper_open = True
                        self.piper_arm.set_gripper(0.08)
                else:
                    if self.gripper_open:
                        self.gripper_open = False
                        self.piper_arm.set_gripper(0.0)

                # 发送位置指令
                self.piper_arm.move_to_pose(
                    target_X, target_Y, target_Z,
                    self.RX, self.RY, self.RZ,
                    speed=50
                )

                # 更新软件位置
                self.X = target_X
                self.Y = target_Y
                self.Z = target_Z

            except Exception as e:
                print(f"[机械臂指令错误] {e}")

    def get_action(self, obs):
        # 3D鼠标优先
        sm_action, is_intervening = self.spacemouse_reader.get_action()
        self.is_intervening = is_intervening

        if is_intervening and sm_action is not None:
            dx = sm_action[0] if abs(sm_action[0]) > self.DEAD_ZONE else 0.0
            dy = sm_action[1] if abs(sm_action[1]) > self.DEAD_ZONE else 0.0
            dz = sm_action[2] if abs(sm_action[2]) > self.DEAD_ZONE else 0.0

            sm_gripper = sm_action[6]
            if abs(sm_gripper) > self.DEAD_ZONE:
                gripper_ctrl = 1.0 if sm_gripper > 0 else -1.0
            else:
                gripper_ctrl = 1.0 if self.gripper_open else -1.0

            override_action = np.array([
                dx,
                dy,
                dz,
                gripper_ctrl
            ], dtype=self._act_spec.dtype)
            override_action = np.clip(override_action, self._act_spec.minimum, self._act_spec.maximum)
            return override_action, True

        # 随机策略
        if self.agent is None or self._global_step < self.seed_steps:
            action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)

            if np.random.random() < self.random_drift_prob or np.linalg.norm(self.last_random_direction) == 0:
                direction = np.random.uniform(-1, 1, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction
            else:
                direction = self.last_random_direction
                direction += np.random.normal(0, 0.15, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction

            action[:3] = direction * self.random_amplitude

            if self._global_step - self.last_gripper_change_step >= self.gripper_change_interval:
                self.random_gripper_state = np.random.choice([1.0, -1.0])
                self.last_gripper_change_step = self._global_step

            action[3] = self.random_gripper_state
            return action, False

        # 智能体策略（no_grad纯读，不需要锁，允许与异步训练并行）
        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self._global_step, eval_mode=False)
        return action, False

    def update_policy(self, num_updates=20):
        if self.agent is None:
            print("⚠️  Agent未初始化，跳过策略更新")
            return
        if self.replay_loader is None:
            print("⚠️  ReplayLoader未初始化，跳过策略更新")
            return

        # 确保replay buffer已加载最新数据
        if hasattr(self, '_replay_buffer') and self._replay_buffer is not None:
            self._replay_buffer._samples_since_last_fetch = self._replay_buffer._fetch_every

        # 检查是否有足够的episode数据
        npz_files = list(self._buffer_dir.glob('*.npz'))
        if len(npz_files) == 0:
            print("⚠️  暂无完整episode数据，跳过策略更新")
            return

        try:
            async_info = f" (异步已更新{self._async_update_count}步)" if self.async_update else ""
            print(f"\n开始同步更新策略，执行 {num_updates} 次梯度更新{async_info}...")
            if self.async_update:
                self._async_pause.set()
                self._async_paused.wait(timeout=5.0)  # 等异步线程确认停下
            for i in range(num_updates):
                metrics = self.agent.update(self.replay_iter, self._global_step)
                if i % 10 == 0:
                    print(f"  进度: {i+1}/{num_updates}", end='\r')
            if self.async_update:
                self._async_pause.clear()
                self._async_paused.clear()
            print(f"\n✅ 同步策略更新完成")
            return metrics
        except Exception as e:
            if self.async_update:
                self._async_pause.clear()
                self._async_paused.clear()
            print(f"❌ 策略更新失败: {e}")
            return None

    def save_snapshot(self):
        snapshot_path = self.output_dir / f'snapshot_robot_{self._global_step}.pt'
        payload = {
            '_global_step': self._global_step,
            '_global_episode': self._global_episode
        }
        if self.agent is not None:
            payload['agent'] = self.agent
        torch.save(payload, snapshot_path)
        print(f"\n✅ 模型保存: {snapshot_path.name}")

    def load_snapshot(self, snapshot_path):
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            print(f"⚠️  快照文件不存在: {snapshot_path}")
            return False

        try:
            payload = torch.load(snapshot_path, map_location=self.device, weights_only=False)
            if 'actor_state_dict' in payload and 'agent' not in payload:
                print("加载预训练Actor权重...")
                if hasattr(self.agent, 'actor'):
                    self.agent.actor.load_state_dict(payload['actor_state_dict'])
                    self.agent.actor.to(self.device)
                    print("✅ Actor权重加载成功")
            else:
                for k, v in payload.items():
                    if k in self.__dict__:
                        self.__dict__[k] = v
                        if k == 'agent' and v is not None:
                            for attr_name in ['encoder', 'actor', 'critic', 'critic_target', 'value_predictor']:
                                if hasattr(v, attr_name):
                                    getattr(v, attr_name).to(self.device)
            print("✅ 快照加载成功")
            return True
        except Exception as e:
            print(f"❌ 加载快照失败: {e}")
            return False

    def visualize(self, obs, reward, episode_step):
        if self._latest_frame is None:
            return True

        frame = self._latest_frame.copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 根据图像尺寸自适应字号
        scale = min(self.IMG_HEIGHT, self.IMG_WIDTH) / 256.0
        y_pos = int(25 * scale)
        line_spacing = int(20 * scale)
        font_title = max(0.5, 0.7 * scale)
        font_body = max(0.35, 0.5 * scale)
        font_small = max(0.3, 0.4 * scale)
        thick = max(1, int(2 * scale))
        thin = max(1, int(1 * scale))

        if self._global_step < self.seed_steps:
            cv2.putText(frame_bgr, "SEEDING...", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_title, (0, 140, 255), thick)
            y_pos += line_spacing
            cv2.putText(frame_bgr, f"Seed: {self._global_step}/{self.seed_steps}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_body, (0, 140, 255), thin)
        else:
            cv2.putText(frame_bgr, "TRAINING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_title, (0, 255, 0), thick)
        y_pos += line_spacing

        if self.is_intervening:
            cv2.putText(frame_bgr, "INTERVENING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_body, (0, 0, 255), thick)
        else:
            cv2.putText(frame_bgr, "Model Control", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_body, (255, 255, 0), thick)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Step: {self._global_step}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_body, (255, 255, 255), thick)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Episode: {self._global_episode + 1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_body, (255, 255, 255), thick)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Reward: {self.episode_reward:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_body, (255, 255, 255), thick)
        y_pos += line_spacing

        # VLM 状态显示
        if self.vlm_rewarder is not None and self.last_vlm_result is not None:
            vlm_stage = self.last_vlm_result.get('stage', 0)
            match_text = f"Stage {vlm_stage}" if self.last_vlm_result.get('is_match') else "Checking"
            confidence = self.last_vlm_result.get('confidence', 0.0)
            match_color = (0, 255, 0) if self.last_vlm_result.get('is_match') else (0, 165, 255)
            cv2.putText(frame_bgr, f"VLM: {match_text} ({confidence:.2f})", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_body, match_color, thick)
            y_pos += line_spacing
            reason = str(self.last_vlm_result.get('reason', ''))
            reason = reason[:35] + '...' if len(reason) > 38 else reason
            cv2.putText(frame_bgr, f"Why: {reason}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (180, 220, 255), thin)
            y_pos += line_spacing

        cv2.putText(frame_bgr, "1=reached 2=grasped 3=lifted s=Save q=Quit", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 200, 255), thin)

        cv2.imshow("Piper Robot Training", frame_bgr)

        # VLM 自动检测（在按键处理之前）
        if self.vlm_rewarder is not None:
            self.last_vlm_result = self.vlm_rewarder.evaluate(
                frame, self.current_stage, self._global_episode, self._global_step,
                camera=self.camera,
                gripper_open=self.gripper_open,
                arm_z=self.Z,
                home_z=self.HOME_Z,
            )
            vlm_matched_stage = self.last_vlm_result.get('stage', self.current_stage)
            if vlm_matched_stage > self.current_stage:
                # VLM判定达到下一阶段，推进分段奖励
                for s in range(self.current_stage + 1, vlm_matched_stage + 1):
                    self.current_stage = s
                    self._reward += self.STAGE_REWARDS[s]
                stage_name = DoubaoGoalRewarder.STAGE_NAMES.get(vlm_matched_stage, '?')
                print(
                    f"[VLM自动奖励] Ep{self._global_episode+1} Step{self._global_step} | "
                    f"stage {vlm_matched_stage}({stage_name}) +{self.STAGE_REWARDS[vlm_matched_stage]:.0f} | "
                    f"conf={self.last_vlm_result['confidence']:.2f} | "
                    f"{self.last_vlm_result['reason']}"
                )

        # 键盘分段奖励
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1') and self.current_stage < 1:
            self.current_stage = 1
            self._reward += self.STAGE_REWARDS[1]
        elif key == ord('2') and self.current_stage < 2:
            self.current_stage = 2
            self._reward += self.STAGE_REWARDS[2]
        elif key == ord('3') and self.current_stage < 3:
            self.current_stage = 3
            self._reward += self.STAGE_REWARDS[3]
        elif key == ord('s'):
            self.save_snapshot()
        elif key == ord('q'):
            return False

        return True

    def pretrain_bc(self, bc_steps=10000, bc_loss_type='mse', log_interval=500):
        """BC pretraining on demo data before RL.

        Only updates encoder + actor. Must be called after agent init
        and before train().
        """
        if self.agent is None:
            print("[BC] WARNING: Agent not initialized, skipping BC pretraining")
            return

        # Check buffer has data
        npz_files = list(self._buffer_dir.glob('*.npz'))
        if len(npz_files) == 0:
            print('[BC] WARNING: No demo data in buffer, skipping BC pretraining')
            return

        buffer_size = len(self.replay_storage)
        print(f"\n{'='*60}")
        print(f"  BC Pretraining: {bc_steps} steps, loss_type={bc_loss_type}")
        print(f"  Demo buffer: {buffer_size} transitions, {len(npz_files)} episodes")
        print(f"{'='*60}")

        # Ensure replay_iter is ready
        if self.replay_iter is None:
            self.replay_iter = iter(self.replay_loader)

        # Force fetch buffer data
        if hasattr(self, '_replay_buffer') and self._replay_buffer is not None:
            self._replay_buffer._samples_since_last_fetch = self._replay_buffer._fetch_every

        for bc_step in range(1, bc_steps + 1):
            try:
                metrics = self.agent.update_bc(self.replay_iter, bc_loss_type)
            except Exception as e:
                print(f"[BC] step {bc_step} error: {e}")
                break

            if bc_step % log_interval == 0 or bc_step == 1:
                print(f"[BC] step {bc_step}/{bc_steps} | "
                      f"bc_loss={metrics.get('bc_loss', 0):.4f} | "
                      f"action_error={metrics.get('bc_action_error', 0):.4f}")

        # Save BC pretrained snapshot
        self.save_snapshot()
        print(f"[BC] Pretraining done.\n")

    def train(self, num_episodes=100, max_steps_per_episode=200):
        print("\n开始训练...")
        print("="*60)

        episodes_bar = tqdm(range(num_episodes), desc="Episodes", unit="episode")

        try:
            for episode in episodes_bar:
                self._global_episode = episode
                self.episode_step = 0
                self.episode_reward = 0.0
                self.current_stage = 0

                # 重置 VLM 检测状态，避免上一 episode 结果污染
                if self.vlm_rewarder is not None:
                    self.vlm_rewarder.reset()
                    self.last_vlm_result = self.vlm_rewarder.last_result

                # 复位
                self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
                self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
                self.gripper_open = True

                with self.arm_command_lock:
                    self.piper_arm.move_to_pose(
                        self.X, self.Y, self.Z,
                        self.RX, self.RY, self.RZ,
                        speed=50
                    )
                    self.piper_arm.set_gripper(0.08)
                time.sleep(2.0)

                # 复位后同步位置
                self._sync_actual_position()

                # 帧初始化：匹配 piper_wrapper.reset() 约定
                # 用 [black, black, first_real_frame] 而非 3 帧真实帧
                self.frames_queue.clear()
                first_frame = self.camera.capture()
                if first_frame is None:
                    first_frame = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8)
                black_frame = np.zeros_like(first_frame)
                self.frames_queue.append(black_frame)
                self.frames_queue.append(black_frame)
                self.frames_queue.append(first_frame)

                # Dummy 首步（匹配 manual_collect.py 和 train_mw.py 的 convention）
                ts_dummy = TimeStep(
                    observation=self.get_stacked_obs(),
                    action=np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype),
                    reward=np.array([0.0], dtype=np.float32),
                    discount=np.array([1.0], dtype=np.float32),
                    first=True,
                    is_last=False,
                    is_intervened=np.array([0.0], dtype=np.float32),
                )
                self.replay_storage.add(ts_dummy)

                obs_prev = self.get_stacked_obs()

                step_bar = tqdm(range(max_steps_per_episode), desc=f"Episode {episode+1}", unit="step", leave=False)
                for step in step_bar:
                    self.episode_step = step
                    self._global_step += 1

                    # 动作
                    action, is_intervened = self.get_action(obs_prev)
                    self._last_action = action
                    self.apply_action(action)

                    # 观测
                    new_frame = self.camera.capture()
                    if new_frame is not None:
                        self.frames_queue.append(new_frame)
                    obs_curr = self.get_stacked_obs()

                    # 可视化（按键奖励在这里产生，必须在奖励读取之前）
                    if not self.visualize(obs_curr, 0.0, step):
                        step_bar.close()
                        episodes_bar.close()
                        print("\n用户退出训练")
                        return

                    # 奖励
                    reward = self._reward
                    if self._reward != 0:
                        self._reward = 0
                    # 非干预帧加时间惩罚，干预帧不加
                    if not is_intervened:
                        reward += self.time_penalty_per_step
                    self.episode_reward += reward

                    # 缓冲区
                    ts = TimeStep(
                        observation=obs_prev,
                        action=action,
                        reward=np.array([reward], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        first=(step == 0),
                        is_last=False,
                        is_intervened=np.array([1.0 if is_intervened else 0.0], dtype=np.float32),
                    )
                    self.replay_storage.add(ts)

                    # 保存
                    if self._global_step - self.last_save_step >= self.save_interval:
                        self.last_save_step = self._global_step
                        self.save_snapshot()

                    obs_prev = obs_curr

                    step_bar.set_postfix({
                        'Global Step': self._global_step,
                        'Reward': f"{self.episode_reward:.1f}",
                        'Intervene': self.is_intervening
                    })

                    # 按3(提起)后提前结束episode
                    if self.current_stage >= 3:
                        break

                step_bar.close()

                # Episode结束
                ts_last = TimeStep(
                    observation=self.get_stacked_obs(),
                    action=self._last_action,
                    reward=np.array([0.0], dtype=np.float32),
                    discount=np.array([0.0], dtype=np.float32),
                    first=False,
                    is_last=True,
                    is_intervened=np.array([0.0], dtype=np.float32),
                )
                self.replay_storage.add(ts_last)

                # 策略更新：seed阶段结束后，每5个episode更新一次
                if (episode + 1) % self.update_every_episodes == 0 and self._global_step >= self.seed_steps:
                    self.update_policy(num_updates=100)
                elif (episode + 1) % self.update_every_episodes == 0 and self._global_step < self.seed_steps:
                    print(f"  Seed阶段未完成 ({self._global_step}/{self.seed_steps})，跳过策略更新")

                episodes_bar.set_postfix({
                    'Last Reward': f"{self.episode_reward:.1f}",
                    'Buffer Size': len(self.replay_storage),
                    'Global Step': self._global_step
                })

        except KeyboardInterrupt:
            print("\n用户中断训练")
        finally:
            episodes_bar.close()
            self.cleanup()

    def cleanup(self):
        print("\n清理资源...")
        self._running = False
        # 异步训练线程是daemon，进程退出自动终止，不需要join
        if self._async_update_count > 0:
            print(f"  异步训练统计: 共 {self._async_update_count} 步异步更新")
        time.sleep(0.5)
        cv2.destroyAllWindows()

        if self.spacemouse_reader is not None:
            self.spacemouse_reader.stop()

        if self.piper_arm is not None:
            try:
                with self.arm_command_lock:
                    self.piper_arm.move_to_pose(
                        self.X, self.Y, self.Z,
                        self.RX, self.RY, self.RZ,
                        speed=20
                    )
            except:
                pass

        if self.camera is not None:
            self.camera.disconnect()

        print("✅ 训练结束，资源已清理")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Piper 机械臂实时训练 (兼容 mentor piper_env)')
    parser.add_argument('--snapshot', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练 episode 数量')
    parser.add_argument('--steps', type=int, default=1500, help='每个 episode 最大步数')
    parser.add_argument('--bc_steps', type=int, default=10000, help='BC预训练步数 (0=不预训练)')
    parser.add_argument('--bc_loss', type=str, default='mse', choices=['mse', 'nll'], help='BC损失类型')
    parser.add_argument('--demo_ratio_start', type=float, default=1.0, help='Demo采样初始比例')
    parser.add_argument('--demo_ratio_end', type=float, default=0.3, help='Demo采样最终比例')
    parser.add_argument('--demo_ratio_decay', type=int, default=50000, help='Demo比例衰减步数')
    parser.add_argument('--no_async', action='store_true', help='禁用异步训练(回退纯同步模式)')
    parser.add_argument('--async_interval', type=float, default=0.01, help='异步更新间隔(秒)')
    parser.add_argument('--negative_ratio', type=float, default=0.2, help='RL阶段负样本采样比例(0=不采样负样本)')
    # VLM 自动奖励
    parser.add_argument('--use_vlm_reward', action='store_true', help='启用基于豆包 VLM 的自动奖励')
    parser.add_argument('--doubao_api_key', type=str, default='', help='豆包 API Key')
    parser.add_argument('--doubao_model', type=str, default='ep-20260606175321-9sh48', help='豆包模型名')
    parser.add_argument('--doubao_base_url', type=str, default='https://ark.cn-beijing.volces.com/api/v3', help='豆包 API Base URL')
    parser.add_argument('--images_dir', type=str, default=str(Path(script_dir) / 'images'),
                        help='三阶段目标图片目录，默认 images/，需包含 target_reached.jpg, target_grasped.jpg, target_lifted.jpg')
    parser.add_argument('--vlm_confidence_threshold', type=float, default=0.60, help='VLM 判定成功所需的最小置信度')
    parser.add_argument('--vlm_check_every', type=int, default=20, help='每隔多少步调用一次 VLM 判定，默认每20步')

    args = parser.parse_args()

    trainer = PiperRobotTrainer()
    trainer.demo_ratio_start = args.demo_ratio_start
    trainer.demo_ratio_end = args.demo_ratio_end
    trainer.demo_ratio_decay_steps = args.demo_ratio_decay
    trainer.negative_ratio = args.negative_ratio
    trainer.async_update = not args.no_async
    trainer.async_update_interval = args.async_interval
    # VLM 自动奖励参数
    trainer.use_vlm_reward = args.use_vlm_reward
    trainer.doubao_api_key = args.doubao_api_key
    trainer.doubao_model = args.doubao_model
    trainer.doubao_base_url = args.doubao_base_url
    trainer.images_dir = args.images_dir
    trainer.vlm_confidence_threshold = args.vlm_confidence_threshold
    trainer.vlm_check_every = args.vlm_check_every
    trainer.init_hardware()

    try:
        from agent_piper import create_piper_agent
        trainer.agent = create_piper_agent(
            obs_shape=trainer._obs_spec.shape,
            action_shape=trainer._act_spec.shape,
            device=trainer.device,
        )
        print("✅ Agent初始化成功")
    except Exception as e:
        import traceback
        print(f"❌ Agent初始化失败:")
        traceback.print_exc()
        trainer.agent = None

    if args.snapshot:
        trainer.load_snapshot(args.snapshot)

    # BC pretraining on demo data
    if args.bc_steps > 0:
        trainer.pretrain_bc(bc_steps=args.bc_steps, bc_loss_type=args.bc_loss)

    trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.steps)


if __name__ == '__main__':
    main()
