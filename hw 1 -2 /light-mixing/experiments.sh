#!/bin/bash

PICO_REMOTE="test"
PICO_LOCAL_1="e6616407e352322c"
PICO_LOCAL_2="e6616407e32e222c"


# REMOTE
Vary target color
python3 sdl_light_mixing.py --pico_id $PICO_REMOTE --color 0 0 0 --save_dir results/remote/vary_target
for r in 25 65; do
    for g in 25 65; do
        for b in 25 65; do
            python3 sdl_light_mixing.py --pico_id $PICO_REMOTE --color $r $g $b --save_dir results/remote/vary_target
        done
    done
done
# Vary random seed
for i in {1..4}; do
    python3 sdl_light_mixing.py --pico_id $PICO_REMOTE --color 25 25 25 --save_dir results/remote/vary_seed
    mv results/remote/vary_seed/R25_G25_B25 results/remote/vary_seed/R25_G25_B25_$i
done



# LOCAL
Vary target color
python3 sdl_light_mixing.py --pico_id $PICO_LOCAL_1 --color 0 0 0 --save_dir results/local/vary_target
for r in 25 65; do
    for g in 25 65; do
        for b in 25 65; do
            python3 sdl_light_mixing.py --pico_id $PICO_LOCAL_1 --color $r $g $b --save_dir results/local/vary_target
        done
    done
done
Vary random seed
for i in {1..4}; do
    python3 sdl_light_mixing.py --pico_id $PICO_LOCAL_1 --color 25 25 25 --save_dir results/local/vary_seed
    mv results/local/vary_seed/R25_G25_B25 results/local/vary_seed/R25_G25_B25_$i
done



# COMBINED
# Same number of iterations
for i in {1..10}; do
    python3 sdl_light_mixing_batch.py --pico_id $PICO_REMOTE $PICO_LOCAL_1 $PICO_LOCAL_2 --color 25 25 25 --num_iter 27 --save_dir results/combined/same_iter
    mv results/combined/same_iter/R25_G25_B25 results/combined/same_iter/R25_G25_B25_$i
done
# Same amount of time
for i in {1..10}; do
    python3 sdl_light_mixing_batch.py --pico_id $PICO_REMOTE $PICO_LOCAL_1 $PICO_LOCAL_2 --color 25 25 25 --num_iter 81 --save_dir results/combined/same_time
    mv results/combined/same_time/R25_G25_B25 results/combined/same_time/R25_G25_B25_$i
done
