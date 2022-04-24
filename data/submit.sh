for dataset in {"SVHN_binary","CIFAR10_binary"}
do
    for train_size in {100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000}
    do
        for seed in {0,16,32,64,128,256,512,1024}
        do
        addqueue -q gpushort -n 1x4 -m 7 -s /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python -m generation.train --dataset_type $dataset --train_dataset_size $train_size --seed $seed --batch_size 256 --lr 0.0002
        done
    done
done


