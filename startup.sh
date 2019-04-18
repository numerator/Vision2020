v4l2-ctl --device=/dev/video0 -c gain_automatic=0 -c white_balance_automatic=0 -c exposure=35 -c gain=0 -c auto_exposure=1 -c brightness=0 -c hue=-32 -c saturation=96

echo "It worked! $(date)" >> "/home/pi/Vision2019/last_run.txt"

killall python

python /home/pi/Vision2019/find_targets.py &
