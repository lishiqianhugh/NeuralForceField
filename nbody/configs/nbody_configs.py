import argparse

def train_arg_parser():
    parser = argparse.ArgumentParser(description="NBody NFF Training")
    
    parser.add_argument('--data_path', type=str, default='nbody.npy', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=501, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--minlr', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--start_time', type=int, default=0, help='Start time index')
    parser.add_argument('--end_time', type=int, default=50, help='End time index')
    parser.add_argument('--layer_num', type=int, default=3, help='Number of layers in ForceFieldPredictor')
    parser.add_argument('--feature_dim', type=int, default=4, help='Feature dimension of the state')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for ForceFieldPredictor')
    parser.add_argument('--step_size', type=float, default=1/200, help='Step size for NeuralODE')
    parser.add_argument('--save_dir', type=str, default='nff', help='Directory to save the training results')
    parser.add_argument('--model_name',type=str,default="nff_model",help='which model to use')
    parser.add_argument('--vis_interval', type=int, default=100, help='Visualization interval')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--sample_num', type=int, default=100, help='Number of samples to train')
    parser.add_argument('--history_len', type=int, default=1, help='history length of slotformer / interaction network')
    parser.add_argument('--num_slots', type=int, default=3, help='Number of slots in SlotFormer')
    

    return parser.parse_args()

def test_arg_parser():
    parser = argparse.ArgumentParser(description="Test a trained model on NBody dataset")

    parser.add_argument('--data_path', type=str, default='nbody.npy', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--start_time', type=int, default=0, help='Start time index')
    parser.add_argument('--end_time', type=int, default=50, help='End time index')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--layer_num', type=int, default=3, help='Number of layers in ForceFieldPredictor')
    parser.add_argument('--feature_dim', type=int, default=4, help='Feature dimension of the state')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for ForceFieldPredictor')
    parser.add_argument('--step_size', type=float, default=1/200, help='Step size for NeuralODE')
    parser.add_argument('--save_dir', type=str, default='nff', help='Directory to save the training results')
    parser.add_argument('--model_name',type=str,default="nff_model",help='which model to use')
    parser.add_argument('--model_path', type=str, default='exps/nff/model_final.pth', help='Path to the trained model')
    parser.add_argument('--sample_num', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--history_len', type=int, default=1, help='history length of slotformer / interaction network')
    parser.add_argument('--num_slots', type=int, default=3, help='Number of slots in SlotFormer')
    

    return parser.parse_args()
