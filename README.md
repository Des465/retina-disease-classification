# Retinal Disease Classification

A multiclass image classifier that identifies glaucoma, diabetic retinopathy, and cataracts from retinal fundus photographs, built on EfficientNet-B3 with transfer learning.


## The problem

Fundus photographs are cheap to take and expensive to read, since interpreting them requires an ophthalmologist. That gap is why automated screening is interesting: not to replace the diagnosis, but to triage which images a specialist should look at first.

The three conditions here don't behave the same way. Diabetic retinopathy leaves fairly distinct lesions. Glaucoma shows up as changes in the optic disc that are subtler and easier to confuse with normal anatomical variation. So the interesting part isn't the headline accuracy - it's where the model fails.

## Approach

- **EfficientNet-B3** pretrained on ImageNet, fine-tuned on the fundus dataset. B3 was a compromise: bigger variants overfit quickly on a dataset this size.
- **Training loop** using Adam, with learning-rate warmup and plateau-based scheduling. Warmup mattered more than I expected, without it the early epochs destabilised the pretrained weights.
- **Augmentation** via Albumentations: colour jitter, Gaussian blur, flips. Fundus images vary a lot in illumination and colour between cameras, so colour jitter is closer to realistic variation than a generic augmentation choice.
- **Class imbalance** handled through augmentation rather than resampling.

## Results

Overall accuracy on a 422-image held-out set: **85.8%**. Macro-averaged F1: **0.856**.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cataract | 0.860 | 0.885 | 0.872 | 104 |
| Diabetic retinopathy | 0.963 | 0.936 | 0.949 | 110 |
| Glaucoma | 0.794 | 0.762 | 0.778 | 101 |
| Normal | 0.811 | 0.841 | 0.826 | 107 |

The per-class breakdown is the table worth reading, and the spread in it is the actual result.

Diabetic retinopathy is comfortably the easiest class at 0.949 - it produces discrete, high-contrast lesions that survive downsampling. Cataracts sit next, since lens opacity changes the whole image rather than one region.

Glaucoma is the weak point, at 0.778 F1 and only 0.762 recall. Nearly a quarter of glaucoma cases are missed. That's the expensive kind of error for a screening tool: a false negative sends someone home. Glaucoma shows up as changes in optic disc cupping, which is a proportional judgement about one small structure, and it overlaps with normal anatomical variation. The confusion is mutual - "normal" has the second-lowest precision at 0.811, meaning a good share of what the model calls healthy isn't.

So the headline 85.8% is close to meaningless on its own. The model is good at the condition that's already easy to spot and mediocre at the one where automated screening would add the most value.

## Limitations

- Public dataset, so the images are cleaner and better-framed than routine clinical photographs would be.
- No external validation set. The numbers describe this dataset and shouldn't be read as clinical performance.
- No calibration work, so the predicted probabilities aren't meaningful as confidence.
- This is a screening-triage experiment, not a diagnostic tool, and shouldn't be used as one.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook retina_classification.ipynb
```


---

Built by Vladimir Tsoy, Applied Mathematics @ UCLA.
