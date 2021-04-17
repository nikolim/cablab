# !/bin/bash
for number in 1 2 3 4 5
do
echo "Run: $number"
python3 train.py -a dqn -n 100
done
