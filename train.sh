
conda activate autoplace

python train.py --nEpochs=50 --nGPU=1 --output_dim=4096 --seqLen=3 --encoder_dim=256 \
                --net=autoplace --logsPath=logs_autoplace --split=val \
                --imgDir='/media/cyw/KESU/datasets/clean_radar/7n5s_xy11_remove_ars548/img' \
                --structDir='/media/cyw/KESU/datasets/clean_radar/7n5s_xy11_remove_ars548/'
