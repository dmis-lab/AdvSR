python train.py Data/Train/iwslt15.cs.en.unigram16k/ --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 --save-dir IWSLT15_CS_EN_INST_ADV_0.25 --max-tokens 4096 --no-epoch-checkpoints --update-freq 8 --num_cands 2 --src_pert_prob 0.25 --tgt_pert_prob 0.25 --adv_sr --sp_model Data/sentencepiece/iwslt15.cs.en.unigram16k/sentencepiece.bpe.model
