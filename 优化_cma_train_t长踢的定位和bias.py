from agent.Base_Agent import Base_Agent as Agent
from math_ops.Math_Ops import Math_Ops as M
from scripts.commons.Script import Script
import numpy as np
from cmaes import CMA
import random

class Kick():
    def __init__(self, script:Script) -> None:
        self.script = script
        a = self.script.args          
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t) # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
        # self.player = Agent(a.i, a.p, a.m, a.u, 3, a.t)
    def kick_once(self, y=0,name="Kick_test"):
        #进行一次踢球,返回球最终坐标
        player = self.player
        #player.path_manager.draw_options(enable_obstacles=True, enable_path=True) # enable drawings of obstacles and path to ball
        behavior = player.behavior
        w = player.world
        r = w.robot
        #设置球场状态与初始化位置
        player.scom.unofficial_set_game_time(0)
        player.scom.unofficial_set_play_mode("PlayOn")
        init_y = random.randint(-5,5)
        player.scom.unofficial_beam((-1,init_y/5,r.beam_height),0)
        player.scom.unofficial_move_ball((0,0,0.042))
        for _ in range(25): 
            player.behavior.execute("Zero_Bent_Knees")
            player.scom.commit_and_send( r.get_command() )
            player.scom.receive()
        #踢球方向
        vec = (1,0)
        #球的最终坐标
        ball_final = [0,0]
        #进行一次踢腿动作
        kick_finished = False
        while not kick_finished:
            ball = w.ball_abs_pos[:2]
            finish = False
            finish = behavior.execute("Basic_Kick", M.vector_angle(vec),False,name)
            player.scom.commit_and_send( r.get_command() ) 
            player.scom.receive()
            if behavior.is_ready("Get_Up") or finish or ball[0] >0.5 or w.time_game > 15.0 :#摔倒/踢腿完成/碰到球但未踢/踢球定位超过15s
                kick_finished = True
                player.scom.unofficial_beam((*r.loc_head_position[0:2],r.beam_height),0)
                behavior.execute_to_completion("Zero_Bent_Knees")
        #等待球完全停止
        ball_stopped = False
        while not ball_stopped:
            behavior.execute("Zero_Bent_Knees")
            #摔倒会导致看不到球，使得球速归零，需保持站立
            if  behavior.is_ready("Get_Up"):
                behavior.execute_to_completion("Get_Up")
            player.scom.commit_and_send( r.get_command() ) 
            player.scom.receive()
            ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
            #踢球完成，记录踢球距离
            if  ball_speed <= 0.02:
                ball_stopped = True
                ball_final = w.ball_abs_pos[:2]
        return ball_final
    

    def set_slot(self,paramlist):
        '''
        设置slot参数
        '''
        slot = self.player.behavior.slot_engine
        #param  = [(220.0, [5, 6, 7, 8, 9, 10, 11], [-10.0, 40.0, 65.0, -60.0, -115.0, 60.0, 0.0]), (100.0, [3, 6, 7, 8, 9, 10], [-45.0, -25.0, 80.0, 0.0, 0.0, 30.0])]
        slot.set_slots("Kick_Motion",paramlist)
    def get_slot(self,name="Kick_Motion"):
        '''
        获取slot参数
        '''
        slot = self.player.behavior.slot_engine
        return slot.get_slots(name)
    
 

pos_str=""
def loss():
    global pos_str 
    # ballpos=[K.kick_once(-1),K.kick_once(-1),K.kick_once(0),K.kick_once(0),K.kick_once(1),K.kick_once(1)]
    ballpos=[K.kick_once(-1),K.kick_once(0),K.kick_once(0),K.kick_once(1)]
    print(f"踢4次的坐标:\n{ballpos}")
    pos_str=f"踢4次的坐标:\n{ballpos}"
    # 计算各个损失值
    loss_values = [((20 - pos[0]) ** 2) * 0.6 + ((pos[1]*3) ** 2) * 0.4 for pos in ballpos]
    print(f"损失值:\n{loss_values}")
    # 对损失值进行排序
    sorted_losses = sorted(loss_values)
    # 取出最低的两个损失值并计算它们的平均值
    lowest_two_losses = sorted_losses[:2]
    average_of_lowest_two = sum(lowest_two_losses) / len(lowest_two_losses)
    print(f"最低两个损失值的平均值:\n{average_of_lowest_two}")
    return average_of_lowest_two
    # return average_loss

import pickle

if __name__ == "__main__":
    '''
    Basic_Kick为读取behavior.slot下各类机器人的动作模板文件，下面为一个简单的CMA优化示例
    安装好相关包后直接运行此文件即可，会输出result记录相关过过程结果
    为提高运行速度，可设定Realtime为off，sync为on加快运行速度
    TODO:
        value值可修改为多次踢球结果的均值，以及对value增加y方向的惩罚
        可对参数进行归一化再传给CMAES进行优化
        可将动作模板分为更多帧，以实现更多非线性运动过程
        增加数据可视化，观察优化效果
    '''

    script = Script()
    K = Kick(script)
    slot = K.player.behavior.slot_engine
    #基础参数
    # ps.这样改后调用的还是kickmotion，但是把"Kick8m_R3"当成初始参数进行优化
    # init_param = K.slot2param(K.get_slot("Kick6m_R3"))
    # print(init_param)
    init_param=slot.get_ball_limits_x_y_smaller("Kick_test")+(slot.get_kick_bias_dir("Kick_test")/100,)
    print(init_param)
    #设定优化器
    try:
        with open('optimizer_state.pkl', 'rb') as f:
            generation_start, optimizer = pickle.load(f)
            generation_start+=1
            print("加载文件进行继续优化")
            print("Resuming from generation:", generation_start)
            with open("result.txt",'a') as f:#记录结果
                f.write(f"\n---------\n加载文件进行继续优化\nLoaded optimizer state from file.\n Resuming from generation:{generation_start} t\n---------\n")
    except Exception:
        print(f"未找到文件，开始重新优化\n{Exception}")
        generation_start=0
        variable_range = 0.2  # 参数的变化范围
        bounds = [
            (max(0, init_param[0] - variable_range), init_param[0] + variable_range),
            (init_param[1] - variable_range, init_param[1] + variable_range),
            (-0.6,0.6),
        ]
        bounds = np.array(bounds)
        optimizer = CMA(mean=np.array(init_param), sigma=0.5,population_size=20,bounds=bounds)
        with open("result.txt",'a') as f:#记录结果
            f.write(f"\n---------\n重新开始优化\n---------\n")
    #优化开始
    count=0
    for generation in range(generation_start,500):
        solutions = []
        for _ in range(optimizer.population_size):
            #获取新参数
            param = optimizer.ask()
            # K.set_slot(K.parma2slot(param))#设置slot新参数

            slot.set_ball_limits_fixed_distance(param[:-1],"Kick_test")
            slot.set_kick_bias_dir(param[-1]*100,"Kick_test")
            # [ 0.17443635 -0.03237196  0.02326123]

            print(f"总参数{param}")
            # value = 15-K.kick_once()[0]#进行一次踢球，返回（15-ballx）
            value =  loss()
            solutions.append((param, value))
            with open("result.txt",'a') as f:#记录结果
                f.write(f"#{generation} {value}\n{param}\n {pos_str}\n")
            print(f"#{generation} {value}")
            print(f"param:{param}")
            # if value < 1:
            #     count+=1
        optimizer.tell(solutions)
        with open('optimizer_state.pkl', 'wb') as f:
            pickle.dump((generation, optimizer), f)
        # 保存后从下一轮开始
            
        # if count > 4:
        #     break
    #输出最小值
    min_value=10000
    with open("result.txt",'r') as f:
        value = []
        lines = f.readlines() 
        for i, line in enumerate(lines):
            if '#' in line:
                v = line.strip().split()[1]
                if float(v) < min_value:
                    min_value=float(v)
                    min_params=lines[i]+lines[i+1]+lines[i+2]+lines[i+3]+lines[i+4]+lines[i+5]+lines[i+6]

        print(f"最小value：{min_value}   参数为{min_params}")
    
    with open("result.txt",'a') as f:#记录结果
        f.write(f"\n最小value：{min_value}\n  参数为{min_params} \n本次训练到此结束")

# 
# Kick8m_R4.xml
# pkl