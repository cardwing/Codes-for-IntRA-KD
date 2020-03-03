python3 -u test_pspnet_multi_scale.py ApolloScape PSPNet train test_img \
                          --lr 0.01 \
                          --gpus 3 4 5 6 7 \
                          --npb \
                          --resume outputs_wo_road03/45_9/_pspnet_model_best.pth.tar \
                          --test_size 1410 \
                          -j 10 \
                          -b 5
