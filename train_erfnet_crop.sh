python3 -u train_erfnet_crop.py ApolloScape PSPNet train val \
                        --lr 0.01 \
                        --gpus 0 1 2 3 \
                        --npb \
                        --resume outputs_erfnet_new/44_36/_pspnet_checkpoint.pth.tar \
                        -j 8 \
                        --epochs 24 \
                        --train_size 1488 \
                        --test_size 1488 \
2>&1|tee train_erfnet_crop_bigger_newest_finetune.log
