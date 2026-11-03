[Bug] fix gemma-2-2b-it-FP8 accuracy #4324
Closed
Closed
[Bug] fix gemma-2-2b-it-FP8 accuracy
#4324
@zhyncs
Description
zhyncs
opened on Mar 11, 2025 · edited by merrymercy
Collaborator
Checklist

1. I have searched related issues but cannot get the expected help.

2. The bug has not been fixed in the latest version.

3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

5. Please use English, otherwise it will be closed.
Describe the bug
The accuracy of neuralmagic/gemma-2-2b-it-FP8 drops from 0.62 to 0.52 in the main branch. It was detected by our nightly CI run. We need to fix this.

neuralmagic/gemma-2-2b-it-FP8 | 0.512 | 0.6
https://github.com/sgl-project/sglang/actions/runs/13800885290

Reproduction
N/A

Environment
N/A