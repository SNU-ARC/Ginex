# Ginex

Ginex is a GNN training system for efficient training of a billion-scale dataset on a single machine by using SSD as a memory extension. Ginex accelerates the entire training procedure by provably optimal in-memory caching of feature vectors which reside on SSD without any negative implication on training quality.

Please refer to the full paper [here](https://link.to.paper).

## Installation and Running a Toy Example

Follow the instructions below to install the requirements and run a toy example using [ogbn_papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) dataset.

### Basic Settings

1. Disable `read_ahead`.
    ```console
    sudo -s
    echo 0 > /sys/block/$block_device_name/queue/read_ahead_kb
    ```

2. Install necessary Linux packages. 
    1. `sudo apt-get install -y build-essential`
    2. `sudo apt-get install -y cgroup-tools`
    3. `sudo apt-get install -y unzip`
    4. `sudo apt-get install -y python3-pip` and `pip3 install --upgrade pip`
    5. Compatible NVIDIA CUDA driver and toolkit. Visit [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for details.

3. Install necessary Python modules. 
    1. PyTorch with version of >= 1.9.0. Visit [here](https://pytorch.org/get-started/locally/) for details.
    2. `pip3 install tqdm`
    3. `pip3 install ogb`
    4. PyG. Visit [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details.

    5. Ninja
        ```console
        sudo wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
        sudo unzip ninja-linux.zip -d /usr/local/bin/
        sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
        ```

4. Use cgroup to mimic the setting where the dataset size is much larger than the main memory size, as assumed in the paper, with ogbn_papers100M dataset. We recommend to limit the memory size to 8GB.
    ```console
    sudo -s
    cgcreate -g memory:8gb
    echo 8000000000 > /sys/fs/cgroup/memory/8gb/memory.limit_in_bytes
    ```

5. Make sure to allocate enough swap space. We recommend to allocate at least 4GB for swap space.
    ```console
    sudo fallocate -l 4G swap.img
    sudo chmod 600 swap.img
    sudo mkswap swap.img
    sudo swapon swap.img
    ```

### Running a toy example

1. Clone our library
    ```console
    git clone https://github.com/SNU-ARC/Ginex.git
    ```
2. Prepare dataset
    ```console
    python3 prepare_dataset.py
    ```
3. Preprocess (Neighbor cache construction)
    ```console
    python3 create_neigh_cache --neigh-cache-size 6000000000
    ````
4. Get `PYTHONPATH`
    ```console
    python3 get_pythonpath.py
    ```
5. Run baseline, i.e., PyG extended to support disk-based processing of graph dataset (denoted as PyG+ in the paper). Replace `PYTHONPATH=...` with the outcome of step 3. `-W ignore` option is used to ignore warnings.
    ```console
    sudo PYTHONPATH=/home/user/.local/lib/python3.8/site-packages cgexec -g memory:8gb python3 -W ignore run_baseline.py
    ```
6. Run Ginex. Replace `PYTHONPATH=...` with the outcome of step 3. `-W ignore` option is used to ignore warnings.
    ```console
    sudo PYTHONPATH=/home/user/.local/lib/python3.8/site-packages cgexec -g memory:8gb python3 -W ignore run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000 --sb-size 1500
    ```

### Results

The following is the result of the toy example on our local server.

#### Environment

* CPU: Intel Xeon Gold 6244 CPU 8-core (16 logical cores with hyper-threading) @ 3.60GHz
* GPU: NVIDIA Tesla V100 16GB PCIe
* Memory: Samsung DDR4-2666 64GB (32GB X 2) (cgroup of 8GB is used)
* Storage: Samsung PM1725b 8TB PCIe Gen3 8-lane
* S/W: Ubuntu 18.04.5 & CUDA 11.4 & Python 3.6.9 & PyTorch 1.9

#### Baseline

Per epoch training time: `216.1687 sec`

#### Ginex

Per epoch training time: `99.5562 sec` (Speedup of 2.2x)

## Maintainer

Yeonhong Park (parkyh96@gmail.com)
Sunhong Min (sunhongmin10@gmail.com)

## Citation

Please cite our paper if you find it useful for your work:

```
@inproceedings{},
  author={},
  booktitle={},
  year={},
```
