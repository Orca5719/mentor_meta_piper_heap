import numpy as np
import time
import os
import cv2
from collections import deque, namedtuple
import sys
import threading
from pathlib import Path
from dm_env import specs

# 确保核心模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage
    from piper_env import PiperRobot
    from realsense_camera import RealSenseCamera
    from spacemouse_reader import SpacemouseReader
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)

# TimeStep定义（支持字符串索引），字段与data_specs的name一一对应
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


class SimpleSpacemouseCollect:
    def __init__(self):
        print("="*60)
        print("     Piper 3D鼠标数据收集工具 (兼容 mentor piper_env)")
        print("="*60)

        self.end = False
        self._running = True
        self._lock = threading.Lock()

        # 硬件对象
        self.camera = None
        self.piper_arm = None
        self.spacemouse_reader = None

        # 图像参数
        self.IMG_HEIGHT = 84   # must match PiperEnv's image_size
        self.IMG_WIDTH = 84    # must match PiperEnv's image_size
        self.frame_stack = 3

        # 机械臂初始位姿 (mm, deg 浮点单位，与 piper_env.py 一致)
        self.HOME_X, self.HOME_Y, self.HOME_Z = 300.614, -12.185, 282.341
        self.HOME_RX, self.HOME_RY, self.HOME_RZ = -179.351, 23.933, 177.934
        self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
        self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
        self.gripper_open = True  # 夹爪状态

        # 数据缓存
        self.frames_queue = deque(maxlen=3)
        self.data_buffer = []  # NPZ备份
        self._buffer_dir = Path.cwd() / 'buffer_robot'
        self.replay_storage = None

        # 状态变量
        self.episode = 0
        self.episode_step = 0
        self._last_action = None
        self._obs_spec = None
        self._act_spec = None
        self._latest_frame = None

        # Spacemouse 参数
        self.DEAD_ZONE = 0.15
        self.SPACE_MOUSE_ACTION_SCALE = 2.0   # mm per action unit, must match PiperEnv's action_scale

        # 控制参数
        self.control_sleep = 0.01

        # Staged reward
        self.STAGE_REWARDS = {1: 4.0, 2: 1.0, 3: 4.0, 4: 1.0}
        self.STAGE_NAMES = {0: "none", 1: "reached", 2: "grasped", 3: "aligned", 4: "placed"}
        self.current_stage = 0

    def _init_specs(self):
        """直接硬编码spec，与 piper_env.py 的 piper_wrapper 输出一致。"""
        # 3帧堆叠 × 3通道 = 9通道, 84×84, uint8
        self._obs_spec = specs.BoundedArray(
            shape=(9, self.IMG_HEIGHT, self.IMG_WIDTH),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        # 4D action: [dx, dy, dz, gripper] ∈ [-1, 1]
        self._act_spec = specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
        print(f"[Spec] Observation: {self._obs_spec.shape}, {self._obs_spec.dtype}")
        print(f"[Spec] Action: {self._act_spec.shape}, {self._act_spec.dtype}")

    def _init_replay_storage(self):
        """初始化ReplayBuffer，spec与 train_mw.py 一致（5个字段，含is_intervened）。"""
        print(f"\n[Buffer] 数据将保存至: {self._buffer_dir.resolve()}")

        self._init_specs()

        # 5个spec，与 train_mw.py 的 data_specs 完全一致
        data_specs = (
            specs.Array(self._obs_spec.shape, self._obs_spec.dtype, 'observation'),
            specs.Array(self._act_spec.shape, self._act_spec.dtype, 'action'),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
            specs.Array((1,), np.float32, 'is_intervened'),
        )

        self._buffer_dir.mkdir(exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)

        print("✅ ReplayBuffer初始化完成")

    def init_hardware(self):
        """初始化硬件，全部使用本地 piper_env.py / realsense_camera.py 接口。"""
        # 相机：使用 RealSenseCamera
        print("\n正在初始化相机...")
        self.camera = RealSenseCamera(
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH)
        )
        self.camera.connect()
        print("✅ 相机初始化成功")

        # 机械臂：使用 piper_env.PiperRobot
        print("正在初始化机械臂...")
        self.piper_arm = PiperRobot("can0")
        self.piper_arm.connect()
        self.piper_arm.enable()
        self.piper_arm.move_to_pose(
            self.HOME_X, self.HOME_Y, self.HOME_Z,
            self.HOME_RX, self.HOME_RY, self.HOME_RZ,
            speed=100
        )
        self.piper_arm.set_gripper(0.08)  # 打开夹爪
        time.sleep(2.0)
        print("✅ 机械臂初始化成功")

        # 初始化Buffer
        self._init_replay_storage()

        # Spacemouse：使用 SpacemouseReader（非阻塞）
        print("正在初始化3D鼠标...")
        self.spacemouse_reader = SpacemouseReader(
            dead_zone=self.DEAD_ZONE,
            action_scale=1.0
        )
        self.spacemouse_reader.start()
        time.sleep(1.0)
        print("✅ 3D鼠标初始化成功")

        # 启动图像采集线程
        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

    def set_gripper_open(self):
        """夹爪打开"""
        self.gripper_open = True
        self.piper_arm.set_gripper(0.08)
        time.sleep(0.5)

    def set_gripper_close(self):
        """夹爪关闭"""
        self.gripper_open = False
        self.piper_arm.set_gripper(0.0)
        time.sleep(0.5)

    def get_stacked_obs(self):
        """获取堆叠帧，严格匹配obs_spec的dtype"""
        with self._lock:
            frames = list(self.frames_queue)

        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))

        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))

        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    def _image_thread(self):
        """图像采集线程"""
        while self._running:
            try:
                frame = self.camera.capture()  # RealSenseCamera.capture() 已 resize，返回 RGB
                if frame is not None:
                    with self._lock:
                        self.frames_queue.append(frame)
                        self._latest_frame = frame.copy()
            except Exception as e:
                print(f"⚠️ 图像采集异常: {e}")
            time.sleep(0.002)

    def align_action(self, dx, dy, dz, gripper_ctrl):
        """将归一化后的控制值对齐到action spec。
        Args:
            dx, dy, dz: 归一化到 [-1, 1] 的位移
            gripper_ctrl: -1.0=关闭, 1.0=打开, 0.0=不变
        """
        action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)
        action[0] = np.clip(dx, -1.0, 1.0)
        action[1] = np.clip(dy, -1.0, 1.0)
        action[2] = np.clip(dz, -1.0, 1.0)
        action[3] = np.clip(gripper_ctrl, -1.0, 1.0)
        return action

    def _reset_arm(self):
        """复位机械臂到初始位姿"""
        self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
        self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
        self.piper_arm.move_to_pose(
            self.X, self.Y, self.Z,
            self.RX, self.RY, self.RZ,
            speed=50
        )
        self.set_gripper_open()
        time.sleep(2.0)

    def _start_new_episode(self):
        """开始新episode：清空帧队列，添加dummy首步。

        匹配 train_mw.py 的 ReplayBuffer convention:
        index 0 = reset observation + action=zeros（dummy），
        index 1 = 第一个真实 action。
        不加 dummy 的话，replay buffer 的 (obs[idx-1], action[idx]) 配对会错位。
        """
        # 清空帧队列，等待新鲜帧
        with self._lock:
            self.frames_queue.clear()
            self._latest_frame = None

        # 等待相机捕获新帧（最多0.5s）
        for _ in range(50):
            with self._lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.01)

        with self._lock:
            if self._latest_frame is not None:
                first_frame = self._latest_frame.copy()
            else:
                first_frame = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8)

        # 构建 initial stacked obs，匹配 piper_wrapper.reset() 约定：[black, black, first_frame]
        black_frame = np.zeros_like(first_frame)
        stacked = np.concatenate([black_frame, black_frame, first_frame], axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        if self._obs_spec.dtype == np.uint8:
            obs_init = stacked.astype(np.uint8)
        else:
            obs_init = (stacked.astype(np.float32) / 255.0)

        # 添加 dummy 首步（匹配 train_mw.py 中 reset 后立即 add 的 convention）
        ts = TimeStep(
            observation=obs_init,
            action=np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype),
            reward=np.array([0.0], dtype=np.float32),
            discount=np.array([1.0], dtype=np.float32),
            first=True,
            is_last=False,
            is_intervened=np.array([1.0], dtype=np.float32),  # demo data
        )
        self.replay_storage.add(ts)

        # 用 [black, first_frame] 初始化帧队列，使后续 stacked obs 自然过渡
        with self._lock:
            self.frames_queue.clear()
            self.frames_queue.append(black_frame)
            self.frames_queue.append(first_frame)

        # 重置 episode 状态
        self.episode_step = 0
        self.end = False
        self.current_stage = 0
        self._last_action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)

    def collect(self, num_episodes=5, max_steps=200, episode_sleep=2.0):
        """主收集循环"""
        self.init_hardware()
        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Data Collection", 640, 480)

        print("\n" + "="*60)
        print("控制说明:")
        print("  3D鼠标移动: 控制机械臂X/Y/Z")
        print("  按钮0: 夹爪关闭 | 按钮1: 夹爪打开")
        print("  1: 到达(+6) | 2: 抓住(+3) | 3: 提起(+1) 并结束episode")
        print("  Q: 退出并保存")
        print("="*60)

        # 添加第一个 episode 的 dummy 首步
        self._start_new_episode()

        try:
            while self.episode < num_episodes and self._running:
                # 1. 读取 Spacemouse（非阻塞）
                sm_action, is_intervening = self.spacemouse_reader.get_action()

                dx_mm, dy_mm, dz_mm = 0.0, 0.0, 0.0
                gripper_ctrl = 0.0

                if is_intervening and sm_action is not None:
                    # SpacemouseReader 输出已归一化到 [-1, 1]
                    dx_mm = sm_action[0] * self.SPACE_MOUSE_ACTION_SCALE  # mm
                    dy_mm = sm_action[1] * self.SPACE_MOUSE_ACTION_SCALE
                    dz_mm = sm_action[2] * self.SPACE_MOUSE_ACTION_SCALE

                    gripper_btn = sm_action[6]
                    if abs(gripper_btn) > 0.1:
                        gripper_ctrl = gripper_btn  # -1=关闭, +1=打开

                # 2. 机械臂物理控制（mm 浮点单位）
                self.X += dx_mm
                self.Y += dy_mm
                self.Z += dz_mm

                # 夹爪
                if gripper_ctrl > 0.1:
                    if not self.gripper_open:
                        self.gripper_open = True
                        self.piper_arm.set_gripper(0.08)
                elif gripper_ctrl < -0.1:
                    if self.gripper_open:
                        self.gripper_open = False
                        self.piper_arm.set_gripper(0.0)

                self.piper_arm.move_to_pose(
                    self.X, self.Y, self.Z,
                    self.RX, self.RY, self.RZ,
                    speed=100
                )

                # 3. 可视化与键盘输入
                current_reward = 0.0
                if self._latest_frame is not None:
                    display = cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(display, f"Ep:{self.episode+1}/{num_episodes} Step:{self.episode_step}/{max_steps}",
                               (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    cv2.putText(display, f"Buf:{len(self.replay_storage)}",
                               (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                    cv2.putText(display, f"Stage:{self.STAGE_NAMES[self.current_stage]}",
                               (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
                    if is_intervening:
                        cv2.putText(display, "INTV", (5, 48),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    cv2.imshow("Data Collection", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('1') and self.current_stage < 1:
                    self.current_stage = 1
                    current_reward = self.STAGE_REWARDS[1]
                    print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | 到达 +{current_reward}")
                elif key == ord('2') and self.current_stage < 2:
                    self.current_stage = 2
                    current_reward = self.STAGE_REWARDS[2]
                    print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | 抓住 +{current_reward}")
                elif key == ord('3') and self.current_stage < 3:
                    self.current_stage = 3
                    current_reward = self.STAGE_REWARDS[3]
                    print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | 对齐 +{current_reward}")
                elif key == ord('4') and self.current_stage < 4:
                    self.current_stage = 4
                    current_reward = self.STAGE_REWARDS[4]
                    self.end = True  # 放置完成 = 成功, 结束episode
                    print(f"[奖励] Ep{self.episode+1} Step{self.episode_step} | 放置 +{current_reward} ★")
                elif key == ord('q'):
                    print("\n用户终止收集，正在保存数据...")
                    self._running = False
                    break

                # 4. 每一步写入TimeStep到Buffer
                if len(self.frames_queue) >= 3:
                    obs_t = self.get_stacked_obs()
                    # 归一化action到 [-1, 1]
                    action_dx = np.clip(dx_mm / self.SPACE_MOUSE_ACTION_SCALE, -1.0, 1.0) if self.SPACE_MOUSE_ACTION_SCALE != 0 else 0.0
                    action_dy = np.clip(dy_mm / self.SPACE_MOUSE_ACTION_SCALE, -1.0, 1.0) if self.SPACE_MOUSE_ACTION_SCALE != 0 else 0.0
                    action_dz = np.clip(dz_mm / self.SPACE_MOUSE_ACTION_SCALE, -1.0, 1.0) if self.SPACE_MOUSE_ACTION_SCALE != 0 else 0.0
                    action_gripper = gripper_ctrl if abs(gripper_ctrl) > 0.1 else (1.0 if self.gripper_open else -1.0)
                    action_t = self.align_action(action_dx, action_dy, action_dz, action_gripper)
                    self._last_action = action_t

                    ts = TimeStep(
                        observation=obs_t,
                        action=action_t,
                        reward=np.array([current_reward], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        first=(self.episode_step == 0),
                        is_last=False,
                        is_intervened=np.array([1.0], dtype=np.float32),  # all demo steps are expert data
                    )
                    self.replay_storage.add(ts)

                    # 同时写入NPZ备份
                    self.data_buffer.append({
                        'observation': obs_t,
                        'action': action_t,
                        'reward': current_reward,
                        'is_intervened': 1.0  # demo data
                    })

                self.episode_step += 1

                # 5. Episode结束
                if self.episode_step >= max_steps or self.end:
                    ts_last = TimeStep(
                        observation=self.get_stacked_obs(),
                        action=self._last_action,
                        reward=np.array([0.0], dtype=np.float32),
                        discount=np.array([0.0], dtype=np.float32),
                        first=False,
                        is_last=True,
                        is_intervened=np.array([1.0], dtype=np.float32),  # demo terminal step
                    )
                    self.replay_storage.add(ts_last)

                    print(f"\n✅ Episode {self.episode+1} 完成，已写入Buffer")
                    print(f"   当前Buffer总步数: {len(self.replay_storage)}")

                    # 机械臂复位
                    self._reset_arm()

                    print(f"请重新摆放物体... 休眠 {episode_sleep}s")
                    time.sleep(episode_sleep)

                    # 开始新episode（含dummy首步）
                    self.episode += 1
                    self._start_new_episode()

                time.sleep(self.control_sleep)

        except KeyboardInterrupt:
            print("\n用户中断收集")
        except Exception as e:
            import traceback
            print(f"\n❌ 收集过程出错: {e}")
            traceback.print_exc()
        finally:
            self._running = False
            time.sleep(0.3)
            cv2.destroyAllWindows()

            if self.spacemouse_reader is not None:
                self.spacemouse_reader.stop()
                print("✅ 3D鼠标已停止")

            if self.piper_arm is not None:
                self.piper_arm.move_to_pose(
                    self.HOME_X, self.HOME_Y, self.HOME_Z,
                    self.HOME_RX, self.HOME_RY, self.HOME_RZ,
                    speed=50
                )
                self.piper_arm.set_gripper(0.08)
                print("✅ 机械臂已回零位")

            if self.camera is not None:
                self.camera.disconnect()
                print("✅ 相机已释放")

            # 保存NPZ备份
            self.save_npz()

            print(f"\n📊 收集完成！Buffer总有效步数: {len(self.replay_storage)}")
            print(f"📂 Buffer文件位置: {self._buffer_dir.resolve()}")

    def save_npz(self, filename="spacemouse_backup.npz"):
        """保存NPZ格式备份"""
        if not self.data_buffer:
            print("⚠️ 没有数据需要备份")
            return

        try:
            observations = np.stack([d['observation'] for d in self.data_buffer])
            actions = np.stack([d['action'] for d in self.data_buffer])
            rewards = np.array([d['reward'] for d in self.data_buffer])
            is_intervened = np.array([d['is_intervened'] for d in self.data_buffer])

            np.savez_compressed(
                filename,
                observations=observations,
                actions=actions,
                rewards=rewards,
                is_intervened=is_intervened
            )
            print(f"✅ NPZ备份已保存: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ NPZ备份失败: {e}")

def main():
    collector = SimpleSpacemouseCollect()
    while True:
        print("\n" + "="*45)
        print("      Piper 数据收集系统")
        print("="*45)
        print(" 1. 开始数据收集")
        print(" 2. 退出")

        choice = input("\n请选择 (1/2): ").strip()
        if choice == '1':
            try:
                num_ep = int(input("收集轮数 [默认5]: ") or "5")
                max_st = int(input("每轮步数 [默认200]: ") or "200")
                ep_slp = float(input("轮间休眠(s) [默认2.0]: ") or "2.0")
                collector.collect(num_episodes=num_ep, max_steps=max_st, episode_sleep=ep_slp)
            except ValueError:
                print("❌ 输入错误，请输入数字")
        elif choice == '2':
            print("退出程序")
            break

if __name__ == '__main__':
    main()
