CUDA_VISIBLE_DEVICES=1 python eval.py \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --model_path log/log_10/ \
    --infos_path log/log_10/infos_aoanet-best.pkl \
    --language_eval 1 \
    --image_root /data/I2T_data/MSCOCO/test2014/ \
    --beam_size 1 \
    --batch_size 100 \
    --split test \
    --secret_size 10

    # CUDA_VISIBLE_DEVICES=1 python eval.py \
    # --dump_images 0 \
    # --dump_json 1 \
    # --num_images -1 \
    # --model_path log/log_20/ \
    # --infos_path log/log_20/infos_aoanet-best.pkl \
    # --language_eval 1 \
    # --image_root /data/I2T_data/MSCOCO/test2014/ \
    # --beam_size 1 \
    # --batch_size 100 \
    # --split test \
    # --secret_size 20

    # CUDA_VISIBLE_DEVICES=1 python eval.py \
    # --dump_images 0 \
    # --dump_json 1 \
    # --num_images -1 \
    # --model_path log/log_30/ \
    # --infos_path log/log_30/infos_aoanet-best.pkl \
    # --language_eval 1 \
    # --image_root /data/I2T_data/MSCOCO/test2014/ \
    # --beam_size 1 \
    # --batch_size 100 \
    # --split test \
    # --secret_size 30

    CUDA_VISIBLE_DEVICES=1 python eval.py \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --model_path log/log_40/ \
    --infos_path log/log_40/infos_aoanet-best.pkl \
    --language_eval 1 \
    --image_root /data/I2T_data/MSCOCO/test2014/ \
    --beam_size 1 \
    --batch_size 100 \
    --split test \
    --secret_size 40

