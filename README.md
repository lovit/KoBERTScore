# Ko-BERTScore

BERTScore using pretrained Korean BERT

## Install

```
git clone https://github.com/lovit/ko-BERTScore
cd ko-BERTScore
python setup.py install
```

## Usage

### Evaluate sentence pairs
```python
from KoBERTScore import BERTScore

model_name = "beomi/kcbert-base"
bertscore = BERTScore(model_name, best_layer=4)

references = [
    '날씨는 좋고 할일은 많고 어우 연휴 끝났다',
    '이 영화 정말 재밌었어요',
    '영화 이야기 하는 문장인데요',
    '이 문장은 점수가 낮아야만 합니다'
]
candidates = [
    '날씨가 좋다 하지만 할일이 많다 일해라 인간',
    '영화 잘 고른거 같아',
    '브라질 열대우림이 장기간 화재로 면적이 줄어들고 있습니다',
    '테넷봤나요? 역의역의역은역인가요?'
]

bertscore(references, candidates)
[0.5643115, 0.4720116, 0.2556618, 0.2268927]
```

### Finding best layer
```
kobertscore best_layer \
  --corpus korsts \
  --model_name_or_path beomi/kcbert-base \
  --draw_plot \
  --plot_path .
```

## Best layer

### KcBERT + KorSTS
- layer index = 4 (0 is token embedding layer)
![](resources/kcbert_korsts.png)
