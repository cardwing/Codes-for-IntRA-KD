python3 -u test_pspnet_multi_scale.py ApolloScape PSPNet train test_img \
                          --lr 0.01 \
                          --gpus 3 4 5 6 7 \
                          --npb \
                          --resume trained_model/pspnet_46_6.pth.tar \
                          --test_size 1410 \
                          -j 10 \
                          -b 5
