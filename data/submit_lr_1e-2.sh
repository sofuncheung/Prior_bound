for model_depth in 2
do    
    for dataset in {"FashionMNIST_binary","KMNIST_binary","EMNIST_binary"}
    do
        for train_size in {100,500,1000,2000,4000,8000,10000,16000}
        do
            for seed in {0,16,32,64,128,256,512,1024}
            do
            addqueue -q gpushort -n 1x4 -m 7 -s /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python -m generation.train --model_depth $model_depth --dataset_type $dataset --train_dataset_size $train_size --seed $seed
            done
        done
    done
done

