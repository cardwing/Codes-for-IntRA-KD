python3 -u train_pspnet_crop_finetune.py ApolloScape PSPNet train val \
                        --lr 0.005 \
                        --gpus 0 1 2 3 4 5 6 7 \
                        --npb \
                        --resume outputs_wo_road03/45_9/_pspnet_model_best.pth.tar \
                        -j 20 \
                        --epochs 24 \
                        --train_size 1380 \
                        --test_size 1380 \
2>&1|tee train_pspnet_finetune_newer.log
