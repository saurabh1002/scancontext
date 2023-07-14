<!-- md preview: Show the rendered HTML markdown to the right of the current editor using ctrl-shift-m.-->

# Working C++ code and Python Bindings to the original ScanContext [repository](https://github.com/irapkaist/scancontext) where only matlab scripts work

## How to install
1. Clone this repository to a local directory
2. run `make install` from within a terminal at the root of this directory

## Usage
1. Run `scan_context_pipeline --help` to know how to use the pipeline
2. You could select an existing dataloader for common datasets like KITTI, Mulran, Apollo or Newer College, or write a new one for any other dataset following the same pattern as in the provided dataloaders
3. The pipeline will save the computed loop closure indices to a file in the dataset root path within the `results` folder
4. If provided with a ground truth closure file, the pipeline will additionally generate a Precision-Recall Table (See the dataloaders for how to provide the ground truth closures)

---------------------------------
# Scan Context

- Scan Context is a global descriptor for LiDAR point cloud, which is proposed in this paper and details are easily summarized in this <a href="https://www.youtube.com/watch?v=_etNafgQXoY"> video </a>.

```
@ARTICLE { gskim-2021-tro,
    AUTHOR = { Giseop Kim and Sunwook Choi and Ayoung Kim },
    TITLE = { Scan Context++: Structural Place Recognition Robust to Rotation and Lateral Variations in Urban Environments },
    JOURNAL = { IEEE Transactions on Robotics },
    YEAR = { 2021 },
    NOTE = { Accepted. To appear. },
}

@INPROCEEDINGS { gkim-2018-iros,
  author = {Kim, Giseop and Kim, Ayoung},
  title = { Scan Context: Egocentric Spatial Descriptor for Place Recognition within {3D} Point Cloud Map },
  booktitle = { Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems },
  year = { 2018 },
  month = { Oct. },
  address = { Madrid }
}
```
- This point cloud descriptor is used for place retrieval problem such as place
recognition and long-term localization.


## What is Scan Context?

- Scan Context is a global descriptor for LiDAR point cloud, which is especially designed for a sparse and noisy point cloud acquired in outdoor environment.
- It encodes egocentric visible information as below:
<p align="center"><img src="example/basic/scmaking.gif" width=400></p>

- A user can vary the resolution of a Scan Context. Below is the example of Scan Contexts' various resolutions for the same point cloud.
<p align="center"><img src="example/basic/various_res.png" width=300></p>


## Acknowledgment
This work is supported by the Korea Agency for Infrastructure Technology Advancement (KAIA) grant funded by the Ministry of Land, Infrastructure and Transport of Korea (19CTAP-C142170-02), and [High-Definition Map Based Precise Vehicle Localization Using Cameras and LIDARs] project funded by NAVER LABS Corporation.


## License
 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

### Copyright
- All codes on this page are copyrighted by KAIST and Naver Labs and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. You must attribute the work in the manner specified by the author. You may not use the work for commercial purposes, and you may only distribute the resulting work under the same license if you alter, transform, or create the work.
