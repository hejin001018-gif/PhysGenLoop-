# Final QA summary

Final status: **PASS** (`reports/quality_audit.json`).

| Sequence | Anomaly | Frames | Onset | SAM2 non-prompt mIoU | Sham PSNR | Sham SSIM | Max displacement |
|---|---|---:|---:|---:|---:|---:|---:|
| soccerball | mid-air hover | 48 | 20 | 0.9229 | 45.42 dB | 0.9977 | 234.24 px |
| drift-straight | instant teleport | 50 | 24 | 0.9653 | 34.73 dB | 0.9867 | 137.98 px |
| car-turn | gravity reversal | 80 | 32 | 0.9753 | 31.66 dB | 0.9884 | 134.40 px |

All three additionally passed:

- identical H.264 codec, 24 fps, 854x480 resolution, and expected frame count for Original/Sham/Anomaly;
- pixel-identical Sham and Anomaly intermediates before the declared onset;
- minimum per-frame Sham SSIM >= 0.95 and mean Sham PSNR >= 30 dB;
- non-zero post-onset visual effect and displacement >= 3% of the frame diagonal;
- no material changes outside the declared repair, anomaly-object, and shadow supports (maximum error <= 1 intensity level);
- decoded encoded-keyframe visual inspection (`reports/encoded_keyframes/`).

These are controlled synthetic training/development samples. Passing this QA does not
turn them into an independent real-world anomaly test set; final benchmark claims still
require a source-disjoint real-anomaly test set.
