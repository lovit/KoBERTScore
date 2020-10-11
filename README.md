# Ko-BERTScore

BERTScore using pretrained Korean BERT

## Install

```
git clone https://github.com/lovit/ko-BERTScore
cd ko-BERTScore
python setup.py install
```

## Usage

### Finding best layer
```
kobertscore best_layer \
  --corpus korsts \
  --model_name_or_path beomi/kcbert-base \
  --draw_plot \
  --plot_path .
```

## Best layer

![](resources/kcbert_korsts.png)
