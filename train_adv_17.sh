# FR EN

make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.25_MIN --max-tokens 4096 --no-epoch-checkpoints \
		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.33_MIN --max-tokens 4096 --no-epoch-checkpoints \
		        --update-freq 8 --num_cands 8 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.3 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.3 --tgt_pert_prob 0.3"


# make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.25_10 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.33_10 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.33 --tgt_pert_prob 0.33"
                
# # AR EN

# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.2 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.2 --tgt_pert_prob 0.2"

# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.2_10 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.2 --tgt_pert_prob 0.2"


# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.25_10 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.33_10 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 10 --src_pert_prob 0.33 --tgt_pert_prob 0.33"
