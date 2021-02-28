#!/bin/bash
mkdir data
gdown "https://drive.google.com/uc?id=1KdpV3M27kV-_QOQOrAentfzZ2tew8YS-&"
unzip -qq images_fullsize.zip -d data
rm -rf images_fullsize.zip