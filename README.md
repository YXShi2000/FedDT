# FedDT: A Flexible Federated Learning Framework in Heterogeneous Environment

This repository is the official implementation of tha paper: FedDT: A Flexible Federated Learning Framework in Heterogeneous Environment.



**Note: We use PyTorch with MPI backend for a Master-Worker computation/communication topology. Therefore, do not use the "pip install" command to install PyTorch!**



Train FedAvg on CIFAR-10

```bash
python run.py     --arch resnet8 --complex_arch master=resnet8,worker=resnet8:lr_s:svm,num_clients_per_model=35 --experiment heterogeneous     --data cifar10 --pin_memory True --batch_size 64 --num_workers 2     --partition_data non_iid_dirichlet --non_iid_alpha 100     --train_data_ratio 1 --val_data_ratio 0.1     --n_clients 105 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 5 --global_n_epochs 40    --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=federated_average  --optimizer adam --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.1     --weight_decay 0 --use_nesterov False --momentum_factor 0     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2020 --pn_normalize False --same_seed_process True
```



Train FedDF on CIFAR-10

```bash
python run.py     --arch svm --complex_arch master=resnet8,worker=resnet8:lr_s:svm,num_clients_per_model=35 --experiment heterogeneous     --data cifar10  --batch_size 32 --num_workers 8     --partition_data non_iid_dirichlet --non_iid_alpha 100     --train_data_ratio 1 --val_data_ratio 0.1     --n_clients 105 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 5  --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=2000,eval_batches_freq=100,early_stopping_server_batches=200     --optimizer adam --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.1     --weight_decay 0 --use_nesterov False --momentum_factor 0     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2020 --pn_normalize False --same_seed_process False --use_hog_feature True  --on_cuda True   --pin_memory False
```



Train FedDT on CIFAR-10 (w/o A-FW)

```bash
python run.py     --arch svm --complex_arch master=resnet8,worker=resnet8:lr_s:svm,num_clients_per_model=35 --experiment heterogeneous     --data cifar10  --batch_size 32 --num_workers 0     --partition_data non_iid_dirichlet --non_iid_alpha 100     --train_data_ratio 1 --val_data_ratio 0.1     --n_clients 105 --participation_ratio 0.1   --n_comm_rounds 100 --local_n_epochs 5  --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=ours_method,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=2000,eval_batches_freq=100,early_stopping_server_batches=200,logits_clean=False     --optimizer adam --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.1     --weight_decay 0 --use_nesterov False --momentum_factor 0     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2020 --pn_normalize False --same_seed_process True --use_hog_feature True  --on_cuda True   --pin_memory False
```



Train FedDT on CIFAR-10 (w A-FW)

```bash
python run.py     --arch svm --complex_arch master=resnet8,worker=resnet8:lr_s:svm,num_clients_per_model=35 --experiment heterogeneous     --data cifar10  --batch_size 32 --num_workers 0     --partition_data non_iid_dirichlet --non_iid_alpha 100     --train_data_ratio 1 --val_data_ratio 0.1     --n_clients 105 --participation_ratio 0.1   --n_comm_rounds 100 --local_n_epochs 5  --world_conf 0,0,1,1,100 --on_cuda True     --fl_aggregate scheme=ours_method,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=2000,eval_batches_freq=100,early_stopping_server_batches=200,logits_clean=True     --optimizer adam --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150     --lr_scheduler MultiStepLR --lr_decay 0.1     --weight_decay 0 --use_nesterov False --momentum_factor 0     --track_time True --display_tracked_time True --python_path /home/dzyao/anaconda3/envs/pytorch/bin/python3.8     --hostfile hostfile     --manual_seed 2020 --pn_normalize False --same_seed_process True --use_hog_feature True  --on_cuda True   --pin_memory False
```

