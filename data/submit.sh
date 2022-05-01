for dataset in {"MNIST_binary","EMNIST_binary","FashionMNIST_binary","KMNIST_binary"}
do
    for train_size in {100,125,158,199,251,316,398,501,630,794,1000,1258,1584,1995,2511,3162,3981,5011,6309,7943,10000}
    do
        for seed in {0,16,32,64,128,256,512,1024}
        do
        addqueue -q gpushort -n 1x4 -m 7 -s /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python -m generation.train --dataset_type $dataset --train_dataset_size $train_size --seed $seed
        done
    done
done


