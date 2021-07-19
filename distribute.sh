#!/bin/sh

#echo "Starting copy to Jetson Nano..."
(rsync -rt --delete . ej:./ai-at-the-edge --exclude '.git'&& echo "Jetson done") &
#echo "Starting copy to Tinker Edge R..."
(rsync -rt --delete . et:./ai-at-the-edge --exclude '.git'&& echo "Tinker done") &
#echo "Starting copy to Raspberry Pi..."
(rsync -rt --delete . ep:./ai-at-the-edge --exclude '.git'&& echo "Raspberry done") &
#echo "Starting copy to Coral Dev Board...--exclude '.git'"
(rsync -rt --delete . ec:./ai-at-the-edge --exclude '.git'&& echo "Coral done") &

for job in `jobs -p`; do
    wait $job
done
