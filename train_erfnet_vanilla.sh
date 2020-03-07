python3 -u train_erfnet_vanilla.py ApolloScape ERFNet train val \
                        --lr 0.01 \
                        --gpus 0 1 2 3 \
                        --npb \
                        --resume path/to/pretrained_model \
                        -j 8 \
                        --epochs 24 \
                        --train_size 1488 \
                        --test_size 1488 \
2>&1|tee train_erfnet.log
