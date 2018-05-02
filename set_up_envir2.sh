#!/bin/bash
chmod 600 /home/guohuajiaohuazi/.kaggle/kaggle.json
kaggle competitions download -c talkingdata-adtracking-fraud-detection -p 'data/'
cd data/
unzip '*.zip'
cd mnt/ssd/kaggle-talkingdata2/competition_files
mv *.csv ../../../../
cd ../../../../
rm -r mnt
cd ..



