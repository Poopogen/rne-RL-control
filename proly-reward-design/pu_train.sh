#!/bin/bash
# train.sh — 1000 epochs, random map per epoch

for i in $(seq 1 100); do
    map=$(( RANDOM % 3 + 1 ))
    echo "=== Epoch $i / 100  |  Map $map ==="
    python -m mlgame3d \
        -i rl_play.py -i hidden -i hidden -i hidden \
        -e 80 \
        -gp items 0 \
        -gp audio false \
        -gp map $map \
        -gp checkpoint 10 \
        -gp max_time 120 \
        Proly.x86_64
done



# #!/bin/bash
# # train.sh — 1000 epochs, cycle maps 1/2/3 every 2 epochs

# for i in $(seq 1 1000); do
#     group=$(( (i - 1) / 2 ))
#     map=$(( group % 3 + 1 ))
#     echo "=== Epoch $i / 1000  |  Map $map ==="
#     python -m mlgame3d \
#         -i rl_play.py -i hidden -i hidden -i hidden \
#         -e 1 \
#         -gp items 0 \
#         -gp audio false \
#         -gp map $map \
#         -gp checkpoint 10 \
#         -gp max_time 120 \
#         -ng \
#         Proly.x86_64
# done