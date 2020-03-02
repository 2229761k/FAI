
# gpuid=0


# nvidia-docker run -v `pwd`:`pwd` -w `pwd` --rm \
#                   --name 'mnist_color_baseline_gpu'${gpuid} \
#                   -e CUDA_VISIBLE_DEVICES=${gpuid} \
#                   -it \
#                   --ipc=host \
#                   feidfoe/pytorch:latest \
# python main.py -e unlearn_0.02  --lr 0.001 --checkpoint unlearn_0.02/checkpoint_step_0255.pth --cuda --data_split test
# python main.py -e unlearn_0.03  --lr 0.001 --checkpoint Ratio_Reverse/checkpoint_step_0049.pth --cuda --data_split test
# python main.py -e unlearn_0.03  --lr 0.001 --checkpoint Ratio_same/checkpoint_step_0049.pth --cuda --data_split test
# python main.py -e unlearn_0.03  --lr 0.001 --checkpoint Ratio_oneone/checkpoint_step_0049.pth --cuda --data_split test


# python main.py -e Ratio_Reverse  --lr 0.001 --checkpoint Ratio_Reverse/checkpoint_step_0049.pth --cuda --data_split test
# python main.py -e Ratio_same  --lr 0.001 --checkpoint Ratio_same/checkpoint_step_0049.pth --cuda --data_split test
# python main.py -e Ratio_oneone  --lr 0.001 --checkpoint Ratio_oneone/checkpoint_step_0049.pth --cuda --data_split test
                  #python main.py -e baseline_0.02  --color_var 0.020 --lr 0.001 --checkpoint checkpoint/baseline_0.02/checkpoint_step_0100.pth --cuda --data_split test


