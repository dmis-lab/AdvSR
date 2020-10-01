# EN CS

make train_adv ARGS="cs.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_CS_EN_ADV_RECHECK_0.33 --max-tokens 4096 --no-epoch-checkpoints \
		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"


# make train_adv ARGS="en.cs.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_EN_CS_ADV_RECHECK_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # FR EN

# make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="fr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_FR_EN_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # EN FR

# make train_adv ARGS="en.fr.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_EN_FR_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="en.fr.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_EN_FR_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # AR EN

# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="ar.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_AR_EN_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # EN AR

# make train_adv ARGS="en.ar.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_EN_AR_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="en.ar.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT17_EN_AR_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # VI EN

# make train_adv ARGS="vi.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_VI_EN_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="vi.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_VI_EN_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # EN VI

# make train_adv ARGS="en.vi.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_EN_VI_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="en.vi.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_EN_VI_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # PL EN

# make train_adv ARGS="pl.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_PL_EN_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="pl.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_PL_EN_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # EN PL

# make train_adv ARGS="en.pl.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_EN_PL_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="en.pl.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_EN_PL_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # TR EN

# make train_adv ARGS="tr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_TR_EN_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="tr.en.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_TR_EN_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

# # EN TR

# make train_adv ARGS="en.tr.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_EN_TR_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25"

# make train_adv ARGS="en.tr.sp16k --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
# 	            --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT13_EN_TR_ADV_0.33 --max-tokens 4096 --no-epoch-checkpoints \
# 		        --update-freq 8 --num_cands 9 --src_pert_prob 0.33 --tgt_pert_prob 0.33"

