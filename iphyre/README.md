# I-PHYRE dataset
* Traininig data: 0-10 games (each with 10 action sequences)
* Within data: 0-10 games (each with 20 action sequences)
* Cross data: 10-40 games (each with 20 action sequences)

# Train

## Train NFF
```
python train.py --dataset_name iphyre --games "range(10)" --sample "[i*10 for i in range(10)]" --save_dir nff --batch_size 6 --vis_interval 3000 --lr 5e-4 --minlr 1e-5 --use_dist_input --num_epochs 3001 --segments "[25]" --model_name nff_model --gamma 1e4 --alpha 0 --beta 0 --dtheta_scale 1e1 --angle_scale 1e1
```

## Train IN
```
python train.py --dataset_name iphyre --games "range(10)" --sample "[i*10 for i in range(10)]" --save_dir in --batch_size 50 --vis_interval 5000 --lr 5e-4 --minlr 1e-5 --use_dist_input --num_epochs 5001 --segments "[25]" --model_name in --gamma 1e4 --alpha 0 --beta 0 --dtheta_scale 1e1 --angle_scale 1e1
```

## Train GCN
```
python train.py --dataset_name iphyre --games "range(10)" --sample "[i*10 for i in range(10)]" --save_dir gcn --batch_size 50 --vis_interval 5000 --lr 5e-4 --minlr 1e-5 --use_dist_input --num_epochs 5001 --segments "[25]" --model_name gcn --gamma 1e4 --alpha 0 --beta 0 --dtheta_scale 1e1 --angle_scale 1e1
```

## Train SlotFormer
```
python train.py --dataset_name iphyre --games "range(10)" --sample "[i*10 for i in range(10)]" --save_dir slotformer --batch_size 50 --vis_interval 5000 --lr 5e-4 --minlr 1e-5 --use_dist_input --num_epochs 5001 --segments "[25]" --model_name slotformer --gamma 1e4 --alpha 0 --beta 0 --dtheta_scale 1e1 --angle_scale 1e1
```

# Test

## Test NFF
```
python test.py --games "range(40)" --sample "[i*5+1 for i in range(20)]" --save_dir nff/test --batch_size 800 --seg 1 --model_path ../checkpoints/iphyre/nff/model_final.pt --visualize_force --model_name nff_model --use_dist_input --history_len 1 --dtheta_scale 1e1 --begin 0 --end 150 --angle_scale 1e1
```
The visualization may take some time.

## Test IN
```
python test.py --games "range(40)" --sample "[i*5+1 for i in range(20)]" --save_dir in/test --batch_size 800 --seg 1 --model_path ../checkpoints/iphyre/in/model_final.pt --visualize_force --model_name in --use_dist_input --history_len 1 --dtheta_scale 1e1 --begin 0 --end 150 --angle_scale 1e1
```

## Test GCN
```
python test.py --games "range(40)" --sample "[i*5+1 for i in range(20)]" --save_dir gcn/test --batch_size 800 --seg 1 --model_path ../checkpoints/iphyre/gcn/model_final.pt --visualize_force --model_name gcn --use_dist_input --history_len 1 --dtheta_scale 1e1 --begin 0 --end 150 --angle_scale 1e1 --hidden_dim 512 --layer_num 5
```

## Test SlotFormer
```
python test.py --games "range(40)" --sample "[i*5+1 for i in range(20)]" --save_dir slotformer/test --batch_size 800 --seg 1 --model_path ../checkpoints/iphyre/slotformer/model_final.pt --visualize_force --model_name slotformer --use_dist_input --history_len 1 --dtheta_scale 1e1 --begin 0 --end 150 --angle_scale 1e1
```

# Plan
```
python planning.py --games "range(3,4)"  --save_dir nff/plan --model_path ../checkpoints/iphyre/nff/model_final.pt --model_name nff_model --use_dist_input --history_len 1 --dtheta_scale 1e1 --angle_scale 1e1 --gamma 1e4 --alpha 0 --lr 5e-5
```

