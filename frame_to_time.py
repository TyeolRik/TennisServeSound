import sys

n_frame = float(sys.argv[1])
# 1 sec : 30 frame = x sec : n_frame 
# x sec = n_frame / 30
time_cost = n_frame / 30.0
print("%s frame costs %0.3lf sec in 30.0 fps" % (sys.argv[1], time_cost))

# time_cost sec : 18.286 ~ 18.743 m = 3600 sec : x m
# x = (18.286 ~ 18.743 * 3600) / time_cost / 1000 = x km/h
speed_per_hour_min = 18.286 * 3600.0 / time_cost / 1000.0
speed_per_hour_max = 18.743 * 3600.0 / time_cost / 1000.0
print("Speed Min : %0.3lf" % speed_per_hour_min)
print("Speed Max : %0.3lf" % speed_per_hour_max)
print("Speed Avg : %0.3lf" % ((speed_per_hour_max + speed_per_hour_min) / 2.0))