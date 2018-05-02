#!/bin/bash
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
source ~/.bashrc
pip install kaggle
pip install lightgbm
sudo apt-get install unzip
mkdir data
rm Anaconda3-5.1.0-Linux-x86_64.sh
mkdir .kaggle

