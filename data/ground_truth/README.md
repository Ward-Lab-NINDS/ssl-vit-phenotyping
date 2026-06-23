# Ground-Truth Data

Keep large images and masks out of normal Git commits. Track only this README, the manifest template, channel metadata template, split files, and small de-identified annotations. Use DVC, Git LFS, or a citable archive for the binary files.

The current channel metadata template records V5 / 647 far red, NWS / 488 green, and T7 / 568 orange as ProCode/readout channels. The likely nucleus channel is structural/reference metadata and should not be used as a ProCode identity channel.

See `docs/ground_truth_data.md` for the full implementation plan.
