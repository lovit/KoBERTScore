models="beomi/kcbert-base monologg/kobert monologg/distilkobert monologg/koelectra-base-v2-discriminator"
for model in ${models}; do
    kobertscore best_layer \
      --corpus korsts \
      --model_name_or_path ${model} \
      --draw_plot \
      --output_dir .
done

