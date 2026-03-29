from random import random

from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Step.Step import Step
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np
from math_ops.Math_Ops import Math_Ops as U
from math_ops.Math_Ops import Math_Ops as M
from behaviors.custom.Step.Step_Generator import Step_Generator

'''
目标：
使用强化学习训练机器人踢球动作
----------
- class Kick: 实现一个OpenAI Gym标准环境，用于训练踢球动作
- class Train: 实现训练新模型或测试现有模型的算法
'''


class Kick(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        # 初始化各种标志和变量
        self.lock_flag = False          # 锁定标志，防止重复操作
        self.sleep = 0                  # 睡眠计数器
        self.reset_time = None          # 重置时间记录
        self.behavior = None            # 行为控制器
        self.path_manager = None        # 路径管理器
        self.bias_dir = None            # 方向偏差补偿
        self.robot_type = r_type        # 机器人类型
        self.kick_ori = 0               # 踢球方向
        self.terminal = False           # 终止状态标志

        # 创建Agent对象: 服务器IP, 代理端口, 监视器端口, 球员号码, 机器人类型, 队名, 开启日志, 开启绘图
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0           # 步数计数器，用于限制回合大小
        self.ball_pos = np.array([0, 0, 0])  # 球的位置

        # 获取Step行为对象，用于控制步态
        self.step_obj: Step = self.player.behavior.get_custom_behavior_object("Step")

        # 根据机器人类型选择关节范围
        # 机器人4：使用2-23号关节
        # 其他机器人：使用2-21号关节
        if self.robot_type == 4:
            self.joint_start = 2   # 起始关节编号
            self.joint_end = 24    # 结束关节编号（不包含）
            self.joint_count = 22  # 关节总数
        else:
            self.joint_start = 2   # 起始关节编号
            self.joint_end = 22    # 结束关节编号（不包含）
            self.joint_count = 20  # 关节总数
        
        # 定义状态空间
        # 基础状态数量 + 关节位置数量 + 关节速度数量 + 球相关状态数量
        base_obs_size = 23  # 步数计数、头部高度、IMU数据、传感器数据等
        ball_obs_size = 8   # 球的相对位置、速度、距离等
        joint_obs_size = self.joint_count * 2  # 关节位置和速度
        
        obs_size = base_obs_size + joint_obs_size + ball_obs_size
        self.obs = np.zeros(obs_size, np.float32)  # 初始化观察向量为0
        self.observation_space = gym.spaces.Box(low=np.full(obs_size, -np.inf, np.float32),
                                                high=np.full(obs_size, np.inf, np.float32), dtype=np.float32)

        # 定义动作空间 - 根据机器人类型决定维度
        MAX = np.finfo(np.float32).max  # float32最大值
        self.no_of_actions = self.joint_count
        self.action_space = gym.spaces.Box(low=np.full(self.no_of_actions, -MAX, np.float32),
                                           high=np.full(self.no_of_actions, MAX, np.float32), dtype=np.float32)

        # 将球放在远处，保持视野中的地标（使用Step行为时，头部会跟随球）
        self.player.scom.unofficial_move_ball((14, 0, 0.042))

        # 设置球的初始位置偏移量
        self.ball_x_center = 0.20       # 球在x方向的偏移中心
        self.ball_y_center = -0.04      # 球在y方向的偏移中心
        
        # 设置比赛模式和初始球位置
        self.player.scom.unofficial_set_play_mode("PlayOn")
        self.player.scom.unofficial_move_ball((0, 0, 0))

        # 加载踢球动作参数
        # 踢球动作名称 - 只使用一种固定的踢球动作
        self.kick_name = "ver_kick"  # 固定使用右腿踢球动作
        # 获取Slot引擎
        self.slot_engine = self.player.behavior.slot_engine
        # 获取踢球动作参数
        self.kick_slots = self.slot_engine.get_slots(self.kick_name)
        
        # 初始化Slot执行参数
        self.slot_number = 0
        self.slot_start_time_ms = 0
        self.slot_start_angles = None
        
        # 预存所有关节的当前位置，用于计算偏移量
        self.base_positions = np.zeros(self.joint_count, np.float32)
    
    def observe(self, init=False):
        """
        观察当前环境状态，构建观察向量
        
        参数:
        init - 是否为初始化观察（重置环境后的第一次观察）
        
        返回:
        obs - 包含环境状态信息的观察向量
        """
        w = self.player.world
        r = self.player.world.robot

        # 如果是初始化观察，重置步数计数器
        if init:
            self.step_counter = 0
            
        # ==================== 基础状态信息 ====================
        # 索引0: 归一化的步数计数
        self.obs[0] = self.step_counter / 20
        
        # 索引1-2: 机器人头部高度及其变化率
        self.obs[1] = r.loc_head_z * 3
        self.obs[2] = r.loc_head_z_vel / 2
        
        # 索引3-4: 机器人躯干的翻滚角和俯仰角
        self.obs[3] = r.imu_torso_roll / 15
        self.obs[4] = r.imu_torso_pitch / 15
        
        # 索引5-7: 陀螺仪数据（角速度）
        self.obs[5:8] = r.gyro / 100
        
        # 索引8-10: 加速度计数据
        self.obs[8:11] = r.acc / 10
        
        # 索引11-16: 左脚传感器数据（位置和力）
        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        
        # 索引17-22: 右脚传感器数据（位置和力）
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        
        # ==================== 关节状态信息 ====================
        # 关节位置信息
        idx = 23  # 当前索引位置
        joint_pos_end = idx + self.joint_count
        self.obs[idx:joint_pos_end] = r.joints_position[self.joint_start:self.joint_end] / 100
        
        # 关节速度信息
        idx = joint_pos_end
        joint_vel_end = idx + self.joint_count
        self.obs[idx:joint_vel_end] = r.joints_speed[self.joint_start:self.joint_end] / 6.1395
        
        # ==================== 球相关信息 ====================
        # 确定球相关信息的观察索引起始位置
        idx = joint_vel_end
        
        # 计算球相对于髋部中心的位置
        ball_rel_hip_center = self.player.inv_kinematics.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        
        # 球位置变化（如果不是初始化且球可见）
        ball_vel_end = idx + 3
        if init:
            self.obs[idx:ball_vel_end] = (0, 0, 0)
        elif w.ball_is_visible:
            self.obs[idx:ball_vel_end] = (ball_rel_hip_center - self.obs[ball_vel_end:ball_vel_end+3]) * 10
            
        # 球相对于髋部中心的位置
        idx = ball_vel_end
        ball_pos_end = idx + 3
        self.obs[idx:ball_pos_end] = ball_rel_hip_center
        
        # 球到髋部中心的距离
        idx = ball_pos_end
        self.obs[idx] = np.linalg.norm(ball_rel_hip_center) * 2
        
        # 归一化的踢球方向与当前机器人方向的差值
        idx = ball_pos_end + 1
        self.obs[idx] = U.normalize_deg(self.kick_ori - r.imu_torso_orientation) / 30

        '''
        步态参数/状态的预期观察（示例）:
        时间步            R  0  1  2  0   1   2   3  4
        进度             1  0 .5  1  0 .25  .5 .75  1
        左腿活动         T  F  F  F  T   T   T   T  T
        参数             A  A  A  B  B   B   B   B  C
        示例说明: (A)步持续时间为3ts, (B)步持续时间为5ts
        '''
        return self.obs

    def sync(self):
        ''' 
        运行单个仿真步骤
        将命令发送到服务器并接收新状态
        '''
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def reset(self):
        """
        重置并稳定机器人
        注意：对某些行为，减少稳定化或添加噪声可能更好
        
        返回:
        obs - 初始观察向量
        """
        # 重置各种标志和计数器
        self.lock_flag = False
        self.player.scom.unofficial_set_play_mode("PlayOn")
        
        # 随机生成球和机器人的初始位置
        Gen_ball_pos = [random() * 5 - 9, random() * 6 - 3, 0]  # x: -9到-4, y: -3到3, z: 0
        Gen_player_pos = (random() * 3 + Gen_ball_pos[0], random() * 3 + Gen_ball_pos[1], 0.5)  # 机器人位置基于球位置随机生成
        
        # 记录球的位置并移动球到指定位置
        self.ball_pos = np.array(Gen_ball_pos)
        self.player.scom.unofficial_move_ball((Gen_ball_pos[0], Gen_ball_pos[1], Gen_ball_pos[2]))
        
        # 重置计数器和行为控制
        self.sleep = 0
        self.step_counter = 0
        self.behavior = self.player.behavior
        r = self.player.world.robot
        w = self.player.world
        t = w.time_local_ms
        self.path_manager = self.player.path_manager
        
        # 获取步态生成器
        gait: Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator
        self.reset_time = t

        # 将机器人浮在空中稳定（连续射束）
        for _ in range(25):
            self.player.scom.unofficial_beam(Gen_player_pos, 0)  # 持续将机器人浮在地面上方
            self.player.behavior.execute("Zero_Bent_Knees")  # 执行"弯曲膝盖"姿势
            self.sync()

        # 将机器人放到地面上
        self.player.scom.unofficial_beam(Gen_player_pos, 0)
        # 移动头部触发物理更新（rcssserver3d在没有关节移动时存在bug）
        r.joints_target_speed[0] = 0.01  
        self.sync()

        # 在地面上稳定机器人
        for _ in range(7):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()
            
        # 走向球 - 主循环，直到机器人正确到达球的位置
        while True and w.time_local_ms - self.reset_time <= 50000:  # 最多等待50秒
            direction = 0  # 初始方向
            
            # 如果机器人摔倒，执行站起动作
            if self.player.behavior.is_ready("Get_Up"):
                self.player.behavior.execute_to_completion("Get_Up")
                
            # 根据机器人类型选择方向偏差值
            self.bias_dir = [0.09, 0.1, 0.14, 0.08, 0.05][r.type]
            biased_dir = M.normalize_deg(direction + self.bias_dir)  # 添加偏差以校正方向
            
            # 计算角度差值（机器人学习使用loc而不是IMU）
            ang_diff = abs(M.normalize_deg(biased_dir - r.loc_torso_orientation))

            # 获取到球的路径
            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)
                
            # 判断是否达到了适合踢球的位置和状态
            if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and  # 球最近可见且角度差小于5度
                    t - w.ball_abs_pos_last_update < 100 and  # 球的绝对位置最近更新过
                    dist_to_final_target < 0.025 and  # 到目标的距离足够近
                    not gait.state_is_left_active and gait.state_current_ts == 2):  # 避免在没有准备和稳定的情况下立即踢球
                break
            else:
                # 继续走向球
                dist = max(0.07, dist_to_final_target)  # 最小距离0.07米
                reset_walk = self.behavior.previous_behavior != "Walk"  # 如果上一个行为不是走路，重置行走行为
                
                # 执行行走行为
                # 参数: 子行为名称, 是否重置, 目标位置, 目标是否为绝对位置, 目标方向, 方向是否为绝对方向, 距离
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)

            self.sync()

        # 记忆变量 - 用于计算奖励
        self.lastx = r.cheat_abs_pos[0]  # 记录机器人初始x位置
        self.act = np.zeros(self.no_of_actions, np.float32)  # 初始化动作为零向量

        return self.observe(True)  # 返回初始观察

    def render(self, mode='human', close=False):
        """
        渲染环境（OpenAI Gym接口要求）
        当前实现为空，不执行任何渲染
        """
        return

    def close(self):
        """
        关闭环境
        清除所有绘图并终止玩家连接
        """
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        """
        执行一步动作，返回新状态、奖励、是否终止和额外信息
        
        参数:
        action - 动作向量，作为关节位置的偏移量
        
        返回:
        obs - 新的观察向量
        reward - 奖励值
        terminal - 是否终止
        {} - 额外信息（当前为空）
        """
        fall=0
        r = self.player.world.robot
        b = self.player.world.ball_abs_pos
        w = self.player.world
        t = w.time_local_ms
        behavior = self.player.behavior
        
        # 指数移动平均，平滑动作变化
        if self.step_counter == 0:
            self.base_positions = r.joints_position[self.joint_start:self.joint_end]
            self.act = action  # 第一步直接使用原始动作
            
            # 重置Slot执行参数
            self.slot_number = 0
            self.slot_start_time_ms = t
            self.slot_start_angles = np.copy(r.joints_position[self.joint_start:self.joint_end])
        else:
            self.act = 0.4 * self.act + 0.6 * action  # 平滑动作变化
        
        # 检查是否有预定义的踢球动作
        if self.kick_slots and self.slot_number < len(self.kick_slots):
            # 当前Slot的执行时间
            elapsed_ms = t - self.slot_start_time_ms
            # 获取当前Slot参数
            delta_ms, indices, angles = self.kick_slots[self.slot_number]
            # print(f"self.slot_number: {self.slot_number}")
            # print(f"delta_ms: {delta_ms}")
            # print(f"elapsed_ms: {elapsed_ms}")
            # 检查是否需要进入下一个Slot
            indices=[idx-2 for idx in indices]
            if elapsed_ms >= delta_ms:
                # 更新开始角度为上一个Slot的目标角度
                self.slot_start_angles[indices] = angles
                
                # 如果还有下一个Slot，则进入下一个
                if self.slot_number + 1 < len(self.kick_slots):
                    self.slot_number += 1
                    elapsed_ms = 0
                    self.slot_start_time_ms = t
                    delta_ms, indices, angles = self.kick_slots[self.slot_number]
                    indices=[idx -2 for idx in indices]
            
            # 计算当前进度
            progress = min(1.0, (elapsed_ms + 20) / delta_ms) if delta_ms > 0 else 1.0
            
            # 计算目标角度
            target_base = self.slot_start_angles.copy()
            # 只对当前Slot指定的关节进行插值
            target_base[indices] = (angles - self.slot_start_angles[indices]) * progress + self.slot_start_angles[indices]
            
            # 提取需要的关节位置作为基础
            self.base_positions = target_base.copy()
            # print(f"self.joint_start: {self.joint_start}")
            # print(f"self.joint_end: {self.joint_end}")
            # print(f"target_base 维度: {target_base.shape}")
            # print(f"self.base_positions 维度: {self.base_positions.shape}")
            # print(f"self.act 维度: {self.act.shape}")
            # print(f"final_positions 维度: {final_positions.shape}")
        else:
            # 如果没有预定义动作或已执行完毕，使用当前关节位置作为基础
            self.base_positions = r.joints_position[self.joint_start:self.joint_end]
        
        # 将模型的动作输出作为偏移量添加到基础位置
        # 缩放偏移量以控制其影响程度
        # 这里作为偏移量的标度因子，可以根据需要调整
        offset_scale = 5.0
        final_positions = self.base_positions + self.act * offset_scale
        
        # 设置关节目标位置
        r.set_joints_target_position_direct(
            slice(self.joint_start, self.joint_end),  # 根据机器人类型选择关节范围
            final_positions,                          # 最终的目标位置
            harmonize=False                           # 不进行协调化，因为每一步都会改变目标
        )
        
        # 设置头部关节位置（0号和1号关节）- 头部俯仰和旋转
        self.player.behavior.head.execute()

        # 执行仿真步骤并更新计数器
        self.sync()
        self.step_counter += 1
        self.lastx = r.cheat_abs_pos[0]

        # 如果slot执行完毕，则终止回合并计算奖励
        if elapsed_ms+20 >= delta_ms and self.slot_number + 1 == len(self.kick_slots):
            # 获取当前观察
            obs = self.observe()
            behavior.execute("Zero_Bent_Knees")
            #摔倒会导致看不到球，使得球速归零，需保持站立
            if  behavior.is_ready("Get_Up"):
                #fall=-2
                behavior.execute_to_completion("Get_Up")
            # 将机器人移动到远处，以便观察球的运动
            self.player.scom.unofficial_beam((-14.5, 0, 0.51), 0)
            
            waiting_steps = 0
            high = 0  # 记录球达到的最大高度
            
            # 等待球停止运动或达到最大等待步数
            while waiting_steps < 550:  # 等待
                # 更新球的最大高度
                if w.ball_cheat_abs_pos[2] > high:
                    high = w.ball_cheat_abs_pos[2]
                self.sync()  # 继续执行仿真步骤
                waiting_steps += 1
                
            # 计算球移动的距离
            dis = np.linalg.norm(self.ball_pos - w.ball_cheat_abs_pos)
            print(f"dis: {dis}")
            print(f"high: {high}")
            
            
            # 奖励计算: 球移动距离 - |球的y偏移| + 10*球的最大高度+fall惩罚
            # 这鼓励球向前移动更远距离，保持方向准确，并适度提高球的高度
            reward = dis*1.5-abs(high-0.4)*10+fall
            self.terminal = True  # 设置终止标志

        else:
            # 如果回合未结束，不给予奖励
            obs = self.observe()
            reward = 0
            self.terminal = False

        return obs, reward, self.terminal, {}


class Train(Train_Base):
    def __init__(self, script) -> None:
        """
        初始化训练类
        
        参数:
        script - 脚本名称，用于记录和日志
        """
        super().__init__(script)

    def train(self, args):
        """
        训练模型
        
        参数:
        args - 训练参数字典
        """
        # --------------------------------------- 学习参数设置
        n_envs = min(16, os.cpu_count())  # 并行环境数量，最多14个或CPU核心数
        n_steps_per_env = 256  # 每个环境的步数（RolloutBuffer大小 = n_steps_per_env * n_envs）
        minibatch_size = 64  # 小批量大小，应该是(n_steps_per_env * n_envs)的因子
        total_steps = 5000000  # 总训练步数
        learning_rate = 3e-4  # 学习率
        folder_name = f'Kick_R{self.robot_type}'  # 模型保存文件夹
        model_path = f'./scripts/gyms/logs/{folder_name}/'  # 模型保存路径

        # --------------------------------------- 运行算法
        def init_env(i_env):
            """
            初始化环境的工厂函数
            
            参数:
            i_env - 环境索引
            
            返回:
            一个返回环境实例的函数
            """
            def thunk():
                return Kick(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        # 创建服务器实例，包括额外的测试服务器
        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)

        # 创建向量化环境
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])  # 训练环境
        eval_env = SubprocVecEnv([init_env(n_envs)])  # 评估环境

        try:
            if "model_file" in args:  # 继续训练现有模型
                model = PPO.load(args["model_file"], env=env, device="cuda", n_envs=n_envs, n_steps=n_steps_per_env,
                                 batch_size=minibatch_size, learning_rate=learning_rate,
                                 gamma=0.99,               # 折扣因子
                                gae_lambda=0.95,          # 广义优势估计的 λ
                                n_epochs=10,              # 训练周期数
                                clip_range=0.2,           # 裁剪范围
                                ent_coef=0.0,             # 熵系数
                                vf_coef=0.5,              # 值函数的损失系数
                                max_grad_norm=0.5,        # 梯度裁剪
                                # lr_schedule='constant',   # 学习率调度器
                                verbose=1)
            else:  # 训练新模型
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size,
                            learning_rate=learning_rate, device="cuda",
                            gamma=0.99,               # 折扣因子
                            gae_lambda=0.95,          # 广义优势估计的 λ
                            n_epochs=10,              # 训练周期数
                            clip_range=0.2,           # 裁剪范围
                            ent_coef=0.0,             # 熵系数
                            vf_coef=0.5,              # 值函数的损失系数
                            max_grad_norm=0.5,        # 梯度裁剪
                            # lr_schedule='constant',   # 学习率调度器
                            )

            # 使用learn_model方法训练模型
            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env,
                                          eval_freq=n_steps_per_env * 20,  # 每20个回合评估一次
                                          save_freq=n_steps_per_env * 20,  # 每20个回合保存一次
                                          backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)  # 等待子进程
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        # 关闭环境和服务器
        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        """
        测试已训练的模型
        
        参数:
        args - 测试参数字典，必须包含model_file和folder_dir
        """
        # 使用不同的服务器和监视器端口
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Kick(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)  # 加载模型

        try:
            # 导出模型为pkl文件以创建自定义行为
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)
            # 测试模型
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        # 关闭环境和服务器
        env.close()
        server.kill()


'''
踢球技能训练说明：

1. 短踢 (Short Kick)：
   - 可调节踢球距离范围：3-9米
   - 目标是精确踢到指定距离
   - 奖励基于球与目标距离的接近程度

2. 长踢 (Long Kick)：
   - 固定为最大力量踢球
   - 目标是踢得尽可能远且精确
   - 平均踢球距离约为19-20米
   - 奖励基于球的行进距离和方向准确性

训练参数：
- 短踢：batch_size=60, n_steps_per_env=120, total_steps=15000000
- 长踢：batch_size=64, n_steps_per_env=128, total_steps=25000000

使用说明：
1. 训练短踢：python -m scripts.commons.Run Basic_Kick
2. 训练长踢：python -m scripts.commons.Run Basic_Kick long_kick=True
3. 测试模型：python -m scripts.commons.Run Basic_Kick model_file=路径 test=True
''' 

''' 
PPO模型参数说明:

model = PPO.load(
                    args["model_file"], 
                    env=env, 
                    device="cpu", 
                    n_envs=n_envs, 
                    n_steps=n_steps_per_env, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate,
                    gamma=0.99,               # 折扣因子
                    gae_lambda=0.95,          # 广义优势估计的 λ
                    n_epochs=10,              # 训练周期数
                    clip_range=0.2,           # 裁剪范围
                    ent_coef=0.0,             # 熵系数
                    vf_coef=0.5,              # 值函数的损失系数
                    max_grad_norm=0.5,        # 梯度裁剪
                    lr_schedule='constant',   # 学习率调度器
                    verbose=1
                )
''' 