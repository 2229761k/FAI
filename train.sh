
# gpuid=0


# # nvidia-docker run -v `pwd`:`pwd` -w `pwd` --rm \
# #                   --name 'mnist_color_baseline_gpu'${gpuid} \
# #                   -e CUDA_VISIBLE_DEVICES=${gpuid} \
# #                   -it \
# #                   --ipc=host \
# #                   feidfoe/pytorch:latest \

# training set bright dark 합친 모든 셋으로 훈련 
# python main.py -e unlearn_0.02 --n_class 1 --lr 0.001  --cuda --is_train --data_split train
# python main.py -e unlearn_0.03 --n_class 1 --lr 0.001  --cuda --is_train --data_split train

# python main.py -e Ratio_oneone --n_class 1 --lr 0.001  --cuda --is_train --data_split train
# python main.py -e Ratio_same --n_class 1 --lr 0.001  --cuda --is_train --data_split train
# python main.py -e Ratio_Reverse --n_class 1 --lr 0.001  --cuda --is_train --data_split train
python main.py -e Ratio_Reverse_wo_GRL --n_class 1 --lr 0.001  --cuda --is_train --data_split train
# python main.py -e Ratio_oneone_wo_GRL --n_class 1 --lr 0.001  --cuda --is_train --data_split train
# python main.py -e Ratio_same_wo_GRL --n_class 1 --lr 0.001  --cuda --is_train --data_split train

