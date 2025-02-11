import argparse
from iphyre.games import PARAS

PARAS_IDX = {key: idx for idx, key in enumerate(PARAS.keys())}

FEATURE_DIM = 9
OBJ_NUM = 12

def planning_arg_parser():
    parser = argparse.ArgumentParser(description='Neural Force Field Planning')
    parser.add_argument('--args_path', type=str, default=None, help='args path, if specified, ignore other args and read from the json file')
    parser.add_argument('--dataset_name', type=str,default="all_prop2", help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--time_steps', type=int, default=150, help='simulation time steps')
    parser.add_argument('--model_name',type=str,default="nff_model",help='which model to use')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--layer_num', type=int, default=3, help='layer number')
    parser.add_argument('--num_epochs', type=int, default=201, help='number of refine epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='refine learning rate')
    parser.add_argument('--save_dir', type=str, default='NFF_continous_t', help='save directory name')
    parser.add_argument('--gamma', type=float, default=1e4, help='weight of mse')
    parser.add_argument('--alpha', type=float, default=1e2, help='weight of res')
    parser.add_argument('--beta', type=float, default=0, help='weight of res res')
    parser.add_argument('--seg', type=int, default="25", help='epoch segments')
    parser.add_argument('--use_adjoint', action='store_true', help='use adjoint method')
    parser.add_argument('--use_dist_mask', action='store_false', help='use dist mask')  # default True
    parser.add_argument('--use_dist_input', action='store_true', help='use dist input')  # default False
    parser.add_argument('--dtheta_scale', type=float, default=1e2, help='dthetadt scale')
    parser.add_argument('--step_size', type=float, default=1/200, help='step size of ode solver')
    parser.add_argument('--games', type=str, default="[0,1,2,3,5,7,9]", help='game ids')
    parser.add_argument('--model_path',type=str,default='./exps/nff_g.pt')
    parser.add_argument('--history_len', type=int, default=1, help='history length of slotformer')
    parser.add_argument('--angle_scale', type=float, default=1e2, help='angle scale in input data')
    parser.add_argument('--eva_sim_number', type=int, default=200, help='angle scale in input data')
    parser.add_argument('--eva_top_number', type=int, default=5, help='angle scale in input data')
    parser.add_argument('--plan_sim_number', type=int, default=5, help='angle scale in input data')
    parser.add_argument('--acceleration_clip', type=float, default=0, help='clip small acceleration to avoid cumulative error')
    return parser.parse_args()

def test_arg_parser():
    parser = argparse.ArgumentParser(description='Neural Force Field Test')
    parser.add_argument('--args_path', type=str, default=None, help='args path, if specified, ignore other args and read from the json file')
    parser.add_argument('--dataset_name', type=str,default="all_prop2", help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--begin', type=int, default=0, help='begin time step')
    parser.add_argument('--end', type=int, default=150, help='end time step')
    parser.add_argument('--model_name',type=str,default="nff_model",help='which model to use')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--layer_num', type=int, default=3, help='layer number')
    parser.add_argument('--save_dir', type=str, default='NFF_continous_t', help='save directory name')
    parser.add_argument('--seg', type=int, default="1", help='epoch segments')
    parser.add_argument('--sample', type=str, default="[10*s for s in range(10)]", help='sample ids')
    parser.add_argument('--games', type=str, default="[0,1,2,3,4,5,7,9]", help='game ids')
    parser.add_argument('--use_adjoint', action='store_true', help='use adjoint method')
    parser.add_argument('--step_size', type=float, default=1/200, help='step size of ode solver')
    parser.add_argument('--model_path',type=str,default='./exps/nff_g.pt')
    parser.add_argument('--visualize_force', action='store_true', help='vis force or trajectory')  # default False
    parser.add_argument('--use_dist_mask', action='store_false', help='use dist mask')  # default True
    parser.add_argument('--use_dist_input', action='store_true', help='use dist input')  # default False
    parser.add_argument('--history_len', type=int, default=1, help='history length of slotformer')
    parser.add_argument('--dist_input_scale', type=float, default=1e2, help='dist input scale')
    parser.add_argument('--dtheta_scale', type=float, default=1e2, help='dthetadt scale')
    parser.add_argument('--angle_scale', type=float, default=1e2, help='angle scale in input data')
    return parser.parse_args()

def train_arg_parser():
    parser = argparse.ArgumentParser(description='Neural Force Field Training')
    parser.add_argument('--args_path', type=str, default=None, help='args path, if specified, ignore other args and read from the json file')
    parser.add_argument('--dataset_name', type=str,default="all_prop2", help='dataset name')
    parser.add_argument('--batch_size', type=int, default=45, help='batch size')
    parser.add_argument('--begin', type=int, default=0, help='begin time step')
    parser.add_argument('--end', type=int, default=150, help='end time step')
    parser.add_argument('--model_name',type=str,default="nff_model",help='which model to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--layer_num', type=int, default=3, help='layer number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--minlr', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=10001, help='number of epochs')
    parser.add_argument('--save_dir', type=str, default='NFF_continous_t_fixrot_180', help='save directory name')
    parser.add_argument('--segments', type=str, default="[25,10]", help='epoch segments')
    parser.add_argument('--sample', type=str, default="[10*s for s in range(10)]", help='sample ids')
    parser.add_argument('--games', type=str, default="[0,1,2,3,4,5,6,7,9]", help='game ids')
    parser.add_argument('--vis_interval', type=int, default=10000, help='visualization interval')
    parser.add_argument('--model_interval', type=int, default=999, help='model save interval')
    parser.add_argument('--use_adjoint', action='store_true', help='use adjoint method')
    parser.add_argument('--gamma', type=float, default=1e4, help='weight of mse')
    parser.add_argument('--alpha', type=float, default=1e2, help='weight of res')
    parser.add_argument('--beta', type=float, default=0, help='weight of res res')
    parser.add_argument('--use_dist_mask', action='store_false', help='use dist mask')  # default True
    parser.add_argument('--use_dist_input', action='store_true', help='use dist input')  # default False
    parser.add_argument('--dist_input_scale', type=float, default=1e2, help='dist input scale')
    parser.add_argument('--dtheta_scale', type=float, default=1e2, help='dthetadt scale')
    parser.add_argument('--angle_scale', type=float, default=1e2, help='angle scale in input data')
    parser.add_argument('--slotres_scale', type=float, default=1e2, help='slotformer prediction res scale')
    parser.add_argument('--step_size', type=float, default=1/200, help='step size of ode solver')
    parser.add_argument('--history_len', type=int, default=1, help='history length of slotformer / interaction network')
    parser.add_argument('--gradient_clip', type=float, default=0.5, help='gradient clip in parameter update')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()

