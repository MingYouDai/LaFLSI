import numpy as np



def system_params():
    params = dict({})


    # mode option


    params['mode'] = 'global'
    params['channel'] = 'multi'
    params['Time_mode'] = 'Auto'
    params['number_of_patch'] = 2
    params['num_of_channel'] = 2
    params['max_channel'] = 3
    params['Filter_channel'] = params['num_of_channel'] * 9

    if params['channel'] == 'multi':
        params['flo_life'] = np.array([140, 490])# 2channel
        params['flo_life_start'] = 30
        params['flo_life_end'] = 630
    params['flo_intensity_ratio'] = np.array([0.7,0.9])
    if params['Time_mode'] == 'Auto':
            params['start'] = 10
            params['end'] = 120
            params['Time_init'] = np.linspace(params['start'],params['end'],params['number_of_patch'])/1000

    # path
    params['dataset_root_dir'] = 'dataset'
    params['dataset_sub_global_dir'] = 'Global_GRR'
    params['sub_dir_train'] = 'train'
    params['sub_dir_eval'] = 'eval'
    params['sub_dir_experiment'] = 'experiment'

    ##############################

    params['global_train_root_path'] =              'dataset/Global_GRR/train/data_batch/'
    params['global_train_gt_path'] =                'dataset/Global_GRR/train/gt'
    params['global_train_BatchCut_path'] =          'dataset/Global_GRR/train/data_batch_cut'
    params['global_train_devide_gt_path'] =         'dataset/Global_GRR/train/devide_gt'
    params['global_train_devide_gt_color_path'] =   'dataset/Global_GRR/train/devide_gt_color'


    params['global_eval_root_path'] =               'dataset/Global_GRR/eval/data_batch/'
    params['global_eval_gt_path'] =                 'dataset/Global_GRR/eval/gt'
    params['global_eval_BatchCut_path'] =           'dataset/Global_GRR/eval/data_batch_cut'
    params['global_eval_devide_gt_path'] =          'dataset/Global_GRR/eval/devide_gt'
    params['global_eval_devide_gt_color_path'] =    'dataset/Global_GRR/eval/devide_gt_color'



    params['global_experiment_root_path'] =         'dataset/Global_GRR/experiment/data_batch/'
    params['global_experiment_BatchCut_path'] =     'dataset/Global_GRR/experiment/data_batch_cut/'

    params['global_experiment_root_path'] =         'test_data'
    params['global_experiment_BatchCut_path'] =     'utils/data_batch_cut/'



    params['result_path'] = 'result'

    params['experiment_cut_idx_path'] ='data/experiment_cut_idx.npy'
    params['train_cut_idx_path'] ='data/train_cut_idx.npy'
    params['eval_cut_idx_path'] ='data/eval_cut_idx.npy'

###########################################

    return params






