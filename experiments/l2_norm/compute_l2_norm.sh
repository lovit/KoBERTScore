korpora lmdata \
  --corpus all \
  --output_dir ./ \
  --sampling_ratio 0.2 \
  --n_first_samples 100000 \
  --min_length 10 \
  --max_length 100

models="beomi/kcbert-base monologg/kobert monologg/distilkobert monologg/koelectra-base-v2-discriminator"
for model in ${models}; do
    kobertscore l2norm \
      --model_name_or_path ${model} \
      --references all.train \
      --output_path ${model}
done

