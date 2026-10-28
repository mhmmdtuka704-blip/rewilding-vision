# Third-party notices and attribution

## SegFormer / pretrained model

The application loads NVIDIA's `nvidia/segformer-b0-finetuned-ade-512-512` through Hugging Face Transformers. Its [model card](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) links the NVIDIA SegFormer license. A complete copy is retained at [licenses/NVIDIA-SegFormer.txt](licenses/NVIDIA-SegFormer.txt).

Those terms limit the work and relevant derivatives to non-commercial research/evaluation. The small RUGD residual adapter is distributed for that stated purpose and does not grant commercial model rights. Base weights are downloaded explicitly from the recorded upstream revision rather than placed in Git history. This repository does not claim ownership of upstream model pretraining.

## RUGD

Credit: Maggie Wigness, Sungmin Eum, John G. Rogers, David Han, Heesung Kwon. *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments*, IROS 2019. Original source: <http://rugd.vision/>.

The recorded acquisition mirror is [WilliamBonilla62/RUGD](https://huggingface.co/datasets/WilliamBonilla62/RUGD), which explicitly grants no new license. Raw imagery and annotation files are not redistributed by this repository. The manifest/provenance identifies the selected public source files and their hashes. Review the original dataset terms before acquisition/use.

## Implementation dependencies

PyTorch, Transformers, NumPy, OpenCV, Pillow, Streamlit, Rasterio/GDAL, Safetensors, PyYAML and other dependencies retain their own licenses. Installing/linking them does not transfer their copyrights to this project. Consult the installed distributions and upstream projects for their applicable notices.

## Company names and visual assets

theOtherDada / tOD is acknowledged as the internship context. Names and trademarks belong to their owners; no company endorsement or trademark license is implied. Corporate logo files and the RUGD photo header used in the original local version are not redistributed. Public artwork in `assets/canopy.svg` and `docs/assets/cover.svg` is code-generated decorative artwork, not field imagery or an accuracy result. Synthetic test fixtures are labeled as such.

## Repository license boundary

See [LICENSE](LICENSE). No repository-wide permissive/commercial license is invented for an internship-origin project with separately governed assets. Copyright holders retain their rights; the public repository is available for review under GitHub's applicable terms. Contact the maintainer to discuss any additional permissions for original project code.
