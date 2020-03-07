python3 -u test_erfnet_multi_scale.py ApolloScape ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 0 1 2 \
                          --npb \
                          --resume trained_model/erfnet_trained.pth.tar \
                          --test_size 1440 \
                          -j 6 \
                          -b 3
