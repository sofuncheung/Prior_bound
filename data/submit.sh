for dataset in {"MNIST_binary","EMNIST_binary","FashionMNIST_binary","KMNIST_binary"}
do
    for train_size in {122,407,1357,4516,15026}
    do
        for seed in {0,16,32,64,128,256,512,1024}
        do
        addqueue -q gpushort -n 1x4 -m 7 -s /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python -m generation.train --dataset_type $dataset --train_dataset_size $train_size --seed $seed --data_seed $seed --normalize_kernel True --compute_prior False
        done
    done
done


