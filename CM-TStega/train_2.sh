id="aoanet"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
CUDA_VISIBLE_DEVICES=0 python train.py --id $id \
    --caption_model aoa \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 4 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json /data/I2T_data/MSCOCO/cocotalk.json \
    --input_label_h5 /data/I2T_data/MSCOCO/cocotalk_label.h5 \
    --input_fc_dir /data/I2T_data/MSCOCO/cocotalk_fc/ \
    --input_att_dir /data/I2T_data/MSCOCO/cocotalk_att \
    --seq_per_img 1 \
    --batch_size 32 \
    --print_steps 1000 \
    --beam_size 1 \
    --learning_rate 2e-4 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_20  \
    $start_from \
    --save_checkpoint_every 12000 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 20 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3 \
    --secret_size 20
