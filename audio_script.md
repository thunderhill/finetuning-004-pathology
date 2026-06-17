# Audio Script — FINETUNING_004 Project Narration
# Target: 2:30 — 3:00 minutes | Voice: conversational, confident, non-technical

---

Imagine a pathologist's AI assistant that works perfectly in the lab where it was trained —
but the moment it's used at a different hospital, with a different staining machine,
its accuracy drops from 97 percent... to 54 percent.

That's not a hypothetical. That's the real-world failure mode that kills pathology AI
before it ever reaches patients.

The problem is colour.

When hospitals prepare tissue samples for microscopy, they dip them in chemical stains
to highlight cell structures. But every lab does this slightly differently —
different reagents, different machines, different protocols.
The same tumour tissue can look completely different in colour depending on where it was processed.

An AI trained at one site learns, without realising it, to rely on that colour signature.
Change the colour — and the model is lost.

Our project targets this failure directly.

We trained a Vision Transformer with 303 million parameters to deliberately ignore colour.
During training, we randomly scrambled the colour profile of every image —
shifting the hues of the stains that pathologists use.
The model never sees the same colour twice, so it's forced to learn
the only thing that doesn't change: the shape and structure of the tissue itself.

The results are striking.

Without this technique, the model drops 44 accuracy points under stain shift.
With it, the same model drops only 13 points —
a 3.4 times smaller degradation, recovering 30 percentage points of real-world accuracy.

We also verified the robustness is genuine.
We hit the model with a completely different set of corruptions it had never seen —
JPEG compression, blur, lower resolution, colour casts.
It dropped only 4.6 points. The robustness generalised beyond what it was trained on.

On the hardware side, we ran everything on a single AMD MI300X —
192 gigabytes of high-bandwidth memory.
Fully training a 303-million-parameter model at large batch sizes
normally requires multiple GPUs or memory-saving workarounds.
The MI300X handles it on one card. We measured 725 training images per second,
with peak memory hitting 90 gigabytes per step —
which would overflow an 80-gigabyte card entirely.

We trained five model variants, compared them head to head, and measured everything.
All reproducible. All on a single GPU.

And we built an interactive demo: upload any histology patch,
get an instant classification across nine tissue types,
and see exactly which part of the image the model relied on — through Grad-CAM overlays.

The bottom line:
A pathology AI that works not just where it was trained, but wherever it's deployed.
Measured, verified, and running on AMD MI300X.

That's what we built.
