#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SRCS=(
    "cs"
)
TGT=en

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=16384
ORIG=$ROOT/iwslt15_orig
DATA=$ROOT/iwslt15.cs.en.sp16k
mkdir -p "$ORIG" "$DATA"

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

URLS=(
    "https://wit3.fbk.eu/archive/2015-01//texts/cs/en/cs-en.tgz"
)
ARCHIVES=(
    "cs-en.tgz"
)
VALID_SETS=(
    "IWSLT15.TED.tst2012.cs-en"
)

# download and extract data
for ((i=0;i<${#URLS[@]};++i)); do
    ARCHIVE=$ORIG/${ARCHIVES[i]}
    if [ -f "$ARCHIVE" ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P "$ORIG" "$URL"
        if [ -f "$ARCHIVE" ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit 1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e "$FILE" ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C "$ORIG" -xzvf "$ARCHIVE"
    fi
done

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "$ORIG/${SRC}-${TGT}/train.tags.${SRC}-${TGT}.${LANG}" \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > "$DATA/train.${SRC}-${TGT}.${LANG}"
    done
done

echo "pre-processing valid data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=${VALID_SETS[i]}
    for FILE in ${VALID_SET[@]}; do
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                > "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done

# learn SPM with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}-${TGT}.${SRC}; echo $DATA/train.${SRC}-${TGT}.${TGT}; done | tr "\n" ",")
echo "learning joint SPM over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.sp \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=unigram

# encode train/valid/test
echo "encoding train/valid with learned SPM"
for SRC in "${SRCS[@]}"; do
    for LANG in "$SRC" "$TGT"; do
        python "$SPM_ENCODE" \
            --model "$DATA/sentencepiece.sp.model" \
            --output_format=piece \
            --inputs $DATA/train.${SRC}-${TGT}.${SRC} $DATA/train.${SRC}-${TGT}.${TGT} \
            --outputs $DATA/train.sp.${SRC}-${TGT}.${SRC} $DATA/train.sp.${SRC}-${TGT}.${TGT} \
            --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
        python "$SPM_ENCODE" \
            --model "$DATA/sentencepiece.sp.model" \
            --output_format=piece \
            --inputs $DATA/valid.${SRC}-${TGT}.${SRC} $DATA/valid.${SRC}-${TGT}.${TGT} \
            --outputs $DATA/valid.sp.${SRC}-${TGT}.${SRC} $DATA/valid.sp.${SRC}-${TGT}.${TGT}
    done
done