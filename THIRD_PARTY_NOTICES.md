# Third-party notices

## FunduSegmenter

- Repository: https://github.com/JusticeZzy/FunduSegmenter (commit `db74034`)
- `fundusegmenter_extension/train_psi.py` is a modified copy of FunduSegmenter's `train.py`.
  Its original header is kept unchanged:

  ```
  MIT License
  Copyright (c) 2025 Zhenyi Zhao
  Copyright (c) University of Dundee
  ```

  All modifications are marked `[Ψ加權]` or `[記憶體]`; the complete diff against the original is
  `fundusegmenter_extension/train_psi.diff`.
- The other scripts in `fundusegmenter_extension/` import FunduSegmenter's `model/` and `utils/`
  at run time and reproduce the logic of its `test.py`; they contain no copied FunduSegmenter code
  beyond the model-construction arguments.
- License note: the FunduSegmenter README states that its code is under the MIT License and that all
  weights (the original RETFound weights and the derived FunduSegmenter weights) are under
  CC BY-NC 4.0; the `LICENSE` file in that repository contains the CC BY-NC 4.0 text. Users of
  `train_psi.py` should check the FunduSegmenter repository for the current terms.
- Required acknowledgment (from the FunduSegmenter README):
  *FunduSegmenter was created by Zhenyi Zhao of the University of Dundee, UK.*
  Citation: Zhao, Z., Mookiah, M. R. K., & Trucco, E. (2026). FunduSegmenter: Leveraging the RETFound
  Foundation Model for Joint Optic Disc and Optic Cup Segmentation in Retinal Fundus Images.
  Translational Vision Science & Technology, 15(5), 14.

## RETFound

- https://github.com/rmaphoh/RETFound — weights under CC BY-NC 4.0 (non-commercial use only).
- No weights are included in this repository.

## psi-weighted-nnunet

- https://github.com/OwenWu930302/psi-weighted-nnunet (Apache-2.0, same author)
- `analysis/refuge_select_proxy.py` imports its `psi/run_all_psi.py` and `psi/recompute_caselevel.py`
  at run time. `analysis/refuge_lists.py` (`select`) and `fundusegmenter_extension/refuge_infer_psi.py`
  (`psi1_convex`, `psi5_exclusion`) contain functions copied verbatim from it.

## REFUGE dataset

- Orlando, J. I. et al. REFUGE Challenge: A unified framework for evaluating automated methods for
  glaucoma assessment from fundus photographs. Medical Image Analysis 59, 101570 (2020).
- No images or annotations are included in this repository; obtain the data from the official source.
