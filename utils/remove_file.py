import os
import shutil


def remove_File(params):
    global_path = []
    global_path.append(params['global_train_root_path'])
    global_path.append(params['global_train_gt_path'])
    global_path.append(params['global_train_devide_gt_path'])
    global_path.append(params['global_train_devide_gt_color_path'])

    global_path.append(params['global_eval_root_path'])
    global_path.append(params['global_eval_gt_path'])
    global_path.append(params['global_eval_devide_gt_path'])
    global_path.append(params['global_eval_devide_gt_color_path'])

    global_path.append(params['global_experiment_BatchCut_path'])

    data_path = []
    data_path.append(params['experiment_cut_idx_path'])
    data_path.append(params['train_cut_idx_path'])
    data_path.append(params['eval_cut_idx_path'])
    data_path.append('data/temp_params_global_experiment.npy')


    for i in range(len(data_path)):
        if os.path.exists(data_path[i]):
            os.remove(data_path[i])


    for i in range(len(global_path)):
        if os.path.exists(global_path[i]):
            shutil.rmtree(global_path[i])




