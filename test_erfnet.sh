python3 -u test_erfnet_new.py ApolloScape PSPNet train test_img \
                          --lr 0.01 \
                          --gpus 0 1 2 \
                          --npb \
                          --resume outputs_erfnet/_pspnet_model_best.pth.tar \
                          --test_size 1488 \
                          -j 6 \
                          -b 3
