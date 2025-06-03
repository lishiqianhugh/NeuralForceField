# N-body dataset
Download our data from the link in the inital README or run `python generate_data.py` to generate data.
* Traininig data: train_data.npy (2 and 3 bodies)
* Within data: within_data.npy (2 and 3 bodies with different initial positions and masses)
* Within data long: within_data_long.npy (2 and 3 bodies with different initial positions and masses and a longer timestep of 150)
* Cross data: cross_data.npy (8 / 10 bodies with a longer timestep of 150)

[sample_num, steps, body_num, feature_dim]

All the data has a zero padding at the end along the dimension of body_num.
# Train
## Train NFF
```
python train.py --save_dir nff --end_time 50 --num_epochs 5001 --model_name nff_model --data_path 'train_data.npy' --minlr 1e-7 --lr 5e-4 --hidden_dim 256 --vis_interval 5000 --batch_size 25 --sample_num 200 --layer_num 3 --method euler --step_size 0.005
```
## Train IN

```
python train.py --save_dir in --end_time 50 --num_epochs 5001 --model_name in --data_path 'train_data.npy' --minlr 1e-7 --lr 5e-4 --hidden_dim 256 --vis_interval 5000 --batch_size 50 --sample_num 200 --layer_num 3 
```
## Train GCN

```
python train.py --save_dir gcn --end_time 50 --num_epochs 5001 --model_name gcn --data_path 'train_data.npy' --minlr 1e-7 --lr 5e-4 --hidden_dim 256 --vis_interval 5000 --batch_size 50 --sample_num 200 --layer_num 3 
```

## Train SlotFormer
```
python train.py --save_dir slotformer --end_time 50 --num_epochs 5001 --model_name slotformer --data_path 'train_data_long.npy' --minlr 1e-7 --lr 5e-4 --hidden_dim 256 --vis_interval 5000 --batch_size 50 --sample_num 200 --layer_num 3 --num_slots 10
```

# Test 

## Test NFF
```
python test.py --save_dir 'nff/test' --end_time 50 --layer_num 3 --model_path '../checkpoints/nbody/nff/model_final.pth' --model_name nff_model --data_path 'within_data.npy' --batch_size 200 --sample_num 200 --method euler --step_size 0.005
```
```
python test.py --save_dir 'nff/test' --end_time 150 --layer_num 3 --model_path '../checkpoints/nbody/nff/model_final.pth' --model_name nff_model --data_path 'cross_data.npy' --batch_size 200 --sample_num 200 --method euler --step_size 0.005
```

## Test IN
```
python test.py --save_dir 'in/test' --end_time 50 --layer_num 3 --model_path '../checkpoints/nbody/in/model_final.pth' --model_name in --data_path 'within_data.npy' --batch_size 200 --sample_num 200
```
```
python test.py --save_dir 'in/test' --end_time 150 --layer_num 3 --model_path '../checkpoints/nbody/in/model_final.pth' --model_name in --data_path 'cross_data.npy' --batch_size 200 --sample_num 200
```

## Test GCN
```
python test.py --save_dir 'gcn/test' --end_time 50 --layer_num 3 --model_path '../checkpoints/nbody/gcn/model_final.pth' --model_name gcn --data_path 'within_data.npy' --batch_size 200 --sample_num 200
```
```
python test.py --save_dir 'gcn/test' --end_time 150 --layer_num 3 --model_path '../checkpoints/nbody/gcn/model_final.pth' --model_name gcn --data_path 'cross_data.npy' --batch_size 200 --sample_num 200
```

## Test SlotFormer
```
python test.py --save_dir 'slotformer/test' --end_time 50 --layer_num 3 --model_path '../checkpoints/nbody/slotformer/model_final.pth' --model_name slotformer --data_path 'within_data.npy' --batch_size 200 --sample_num 200 --num_slots 10
```
```
python test.py --save_dir 'slotformer/test' --end_time 150 --layer_num 3 --model_path '../checkpoints/nbody/slotformer/model_final.pth' --model_name slotformer --data_path 'cross_data.npy' --batch_size 200 --sample_num 200 --num_slots 10
```

# Plan

```
python planning.py --save_dir 'nff/plan' --end_time 50 --layer_num 3 --model_path '../checkpoints/nbody/nff/model_final.pth' --model_name nff_model --data_path 'within_data.npy' --batch_size 200 --sample_num 200
```
