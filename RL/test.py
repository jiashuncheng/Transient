import os
import json

#TODO - watermaze
#LINK - /home/jiashuncheng/code/Trasient/RL/train_dir/watermaze_mg/20240125_Watermaze2d_hard_v10_gtrnn_alpha0.98_gate0_0.1_3.0_region0_lr_1e-3_seed1_size11_sigm1_300/.summary/0/events.out.tfevents.1706409390.titanX
if 0:
    file = ["/home/jiashuncheng/code/Trasient/RL/train_dir/watermaze_mg/20240125_Watermaze2d_hard_v10_gtrnn_alpha0.98_gate0_0.1_3.0_region0_lr_1e-3_seed1_size11_sigm1_300/.summary/0/events.out.tfevents.1706409390.titanX"]
    for i in range(len(file)):
        cmd1 = "python"
        cmd2 = "/home/jiashuncheng/code/Trasient/RL/examples/watermaze2d/enjoy_watermaze2d.py"
        exper = file[i].split(".summary")[0]
        with open(exper + "config.json") as user_file:
            parsed_json = json.load(user_file)
        cmd3 = parsed_json["command_line"]
        os.system("nohup "+cmd1+" -u "+cmd2+" "+cmd3+" --no_render  > /dev/null 2>&1 &")
        # os.system(cmd1+" "+cmd2+" "+cmd3+" --no_render")

#TODO - TI df
if 1:
    file="/home/jiashuncheng/code/Trasient/RL/examples/direction_following/enjoy_df.py"
    list=(
"/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240115_MortarMayhemOA_v13_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_distribution0_lr2e-4_seed1_sigm1_600/.summary/0/events.out.tfevents.1706938444.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240115_MortarMayhemOA_v13_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_distribution0_lr2e-4_seed2_sigm1_600/.summary/0/events.out.tfevents.1706938445.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240115_MortarMayhemOA_v13_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_distribution0_lr2e-4_seed5_sigm1_600/.summary/0/events.out.tfevents.1706938443.amax",
) + \
    ("/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240116_MortarMayhemOA_v19_myrnn_alpha0.98_gate0_0.1__distribution0_lr2e-4_seed6_600/.summary/0/events.out.tfevents.1706955952.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240116_MortarMayhemOA_v19_myrnn_alpha0.98_gate0_0.1__distribution0_lr2e-4_seed3_600/.summary/0/events.out.tfevents.1706938446.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/df_mg_600/20240116_MortarMayhemOA_v19_myrnn_alpha0.98_gate0_0.1__distribution0_lr2e-4_seed5_600/.summary/0/events.out.tfevents.1706955951.amax",
)
    for i in range(len(list)):
        cmd1 = "python"
        cmd2 = file
        exper = list[i].split(".summary")[0]
        with open(exper + "config.json") as user_file:
            parsed_json = json.load(user_file)
        cmd3 = parsed_json["command_line"]
        os.system(cmd1+" "+cmd2+" "+cmd3+" --no_render")

#TODO - TI mdf
if 0:
    file="/home/jiashuncheng/code/Trasient/RL/examples/multi_direction_following/enjoy_mdf.py"
    list=("/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_region1_comm6_lr2e-4_seed6_vm10.0_vgamma2.0_uniform_900_pos_init/.summary/0/events.out.tfevents.1706700015.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_region1_comm6_lr2e-4_seed4_vm10.0_vgamma2.0_uniform_900_pos_init/.summary/0/events.out.tfevents.1706700014.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_region1_comm6_lr2e-4_seed1_vm10.0_vgamma2.0_uniform_900_pos_init/.summary/0/events.out.tfevents.1706607943.amax") + \
    ("/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate0_0.1__region0_comm6_lr2e-4_seed1_900/.summary/0/events.out.tfevents.1706535747.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate0_0.1__region0_comm6_lr2e-4_seed2_900/.summary/0/events.out.tfevents.1706535748.amax",
"/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate0_0.1__region0_comm6_lr2e-4_seed3_900/.summary/0/events.out.tfevents.1706535749.amax")
    for i in range(len(list)):
        cmd1 = "python"
        cmd2 = file
        exper = list[i].split(".summary")[0]
        with open(exper + "config.json") as user_file:
            parsed_json = json.load(user_file)
        cmd3 = parsed_json["command_line"]
        os.system(cmd1+" "+cmd2+" "+cmd3+" --no_render")

#TODO - success mdf
#LINK - /home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate0_0.1__region0_comm6_lr2e-4_seed3_900/.summary/0/events.out.tfevents.1706535749.amax
if 0:
    file="examples/multi_direction_following/enjoy_mdf.py"
    list=["/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate1_0.1_vm10.0_vgamma2.0_region1_comm6_lr2e-4_seed6_vm10.0_vgamma2.0_uniform_900_pos_init/.summary/0/events.out.tfevents.1706700015.amax"]+\
    ["/home/jiashuncheng/code/Trasient/RL/train_dir/mdf_mg/20240116_MortarMayhem_v20_gtrnn_alpha0.98_gate0_0.1__region0_comm6_lr2e-4_seed3_900/.summary/0/events.out.tfevents.1706535749.amax"]
    print(list)
    for i in range(len(list)):
        cmd1 = "python"
        cmd2 = file
        exper = list[i].split(".summary")[0]
        with open(exper + "config.json") as user_file:
            parsed_json = json.load(user_file)
        cmd3 = parsed_json["command_line"]
        os.system(cmd1+" "+cmd2+" "+cmd3+" --no_render")