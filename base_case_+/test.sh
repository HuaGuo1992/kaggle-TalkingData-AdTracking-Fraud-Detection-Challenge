#!/bin/bash
python test.py >> test.log
python test1.py >> test1.log
cd ../test
python test.py >> test.log 
