The following commands are to be run from the directory containing the hw1 folder (16831-S25-HW if cloned from GitHub):

In order to run analysis for Question 1 Part 2:

>> python3 hw1/rob831/infrastructure/dataset_inspect.py EXPERT_DATASET_PATH

Note: replace the expert dataset path with the true path of the file, and the mean returns and STD will be printed to the terminal.

------------------------------------------------------------------------------------------------------------

In order to run BC for Question 1 Part 3...

1) For the Ant-v2 environment:

>> python3 hw1/rob831/scripts/run_hw1.py --env_name Ant-v2 \
    --expert_policy_file hw1/rob831/policies/experts/Ant.pkl \
    --exp_name bc_ant --n_iter 1 --video_log_freq -1 \
    --n_layers 3 --size 64 --learning_rate 5e-3 \
    --expert_data hw1/rob831/expert_data/expert_data_Ant-v2.pkl \
    --num_agent_train_steps_per_iter 1000 --eval_batch_size 15000 \
    --train_batch_size 100


2) For the Humanoid-v2 environment:

>> python3 hw1/rob831/scripts/run_hw1.py --env_name Humanoid-v2 \
    --expert_policy_file hw1/rob831/policies/experts/Humanoid.pkl \
    --exp_name bc_humanoid --n_iter 1 --video_log_freq -1 \
    --n_layers 3 --size 64 --learning_rate 5e-3 \
    --expert_data hw1/rob831/expert_data/expert_data_Humanoid-v2.pkl \
    --num_agent_train_steps_per_iter 1000 --eval_batch_size 15000 \
    --train_batch_size 100

------------------------------------------------------------------------------------------------------------

In order to run BC for Question 1 Section 4 to evaluate impact of train batch size on returns...

>> python3 hw1/rob831/scripts/run_hw1.py --env_name Ant-v2 \
    --expert_policy_file hw1/rob831/policies/experts/Ant.pkl \
    --exp_name bc_ant --n_iter 1 --video_log_freq -1 \
    --n_layers 3 --size 64 --learning_rate 5e-3 \
    --expert_data hw1/rob831/expert_data/expert_data_Ant-v2.pkl \
    --num_agent_train_steps_per_iter 1000 --eval_batch_size 15000 \
    --train_batch_size X

Where X in [1,10,50,100,500,1000,2000]

------------------------------------------------------------------------------------------------------------

In order to run DAgger for Question 2 Part 2...

1) For the Ant-v2 environment:

>> python3 hw1/rob831/scripts/run_hw1.py --env_name Ant-v2 \
    --expert_policy_file hw1/rob831/policies/experts/Ant.pkl \
    --exp_name dagger_ant --n_iter 5 --do_dagger --video_log_freq -1 \
    --n_layers 3 --size 64 --learning_rate 5e-3 \
    --expert_data hw1/rob831/expert_data/expert_data_Ant-v2.pkl \
    --num_agent_train_steps_per_iter 1000 --eval_batch_size 15000 \
    --train_batch_size 100


2) For the Humanoid-v2 environment:

>> python3 hw1/rob831/scripts/run_hw1.py --env_name Humanoid-v2 \
    --expert_policy_file hw1/rob831/policies/experts/Humanoid.pkl \
    --exp_name dagger_humanoid --n_iter 30 --do_dagger --video_log_freq -1 \
    --n_layers 4 --size 128 --learning_rate 5e-4 \
    --expert_data hw1/rob831/expert_data/expert_data_Humanoid-v2.pkl \
    --num_agent_train_steps_per_iter 1000 --eval_batch_size 15000 \
    --train_batch_size 100

------------------------------------------------------------------------------------------------------------
