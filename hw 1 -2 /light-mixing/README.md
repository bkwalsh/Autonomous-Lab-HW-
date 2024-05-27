# Light Mixing

[Youtube tutorial](https://www.youtube.com/watch?v=D54yfxRSY6s&t=557s)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sparks-baird/self-driving-lab-demo/blob/main/notebooks/4.2-paho-mqtt-colab-sdl-demo-test.ipynb)

See `sdl_light_mixing_demo.ipynb` for a concise tutorial. To run all experiments in part 1, 2, 3, run the bash script `experiments.sh`.

## Part 1: Remote Light Mixing

Run experiments on the color-mixing SDL-Demo hardware located at the University of Utah. We run eight experiments, each with a different color combination. The experiment code is the same as in `sdl_light_mixing_demo.ipynb`. Results are found in `results/remote`. We tested varying the target color and the random seed. Results are found in `results/remote/vary_target` and `results/remote/vary_seed`, respectively.

Usage: `python3 sdl_light_mixing.py --pico_id test --color <R> <G> <B> --save_dir <save_dir>`

Example: `python3 sdl_light_mixing.py --pico_id test --color 25 25 25 --save_dir results/remote/vary_target`

## Part 2: Local Light Mixing

Run experiments on the color-mixing SDL-Demo local Pico W microcontroller [setup](https://projects.raspberrypi.org/en/projects/get-started-pico-w/1). We run the same eight experiments as in part 1. Results are found in `results/local/vary_target` and `results/local/vary_seed`, respectively.

Usage: `python3 sdl_light_mixing.py --pico_id <pico_id> --color <R> <G> <B> --save_dir <save_dir>`

Example: `python3 sdl_light_mixing.py --pico_id e6616407e352322c e6616407e32e222c --color 25 25 25 --save_dir results/local/vary_target`

## Part 3: Combining Devices

Batch optimization with multiple devices, following this [example](https://self-driving-lab-demo.readthedocs.io/en/latest/notebooks/6.3-batch-optimization.html). We use two local Pico W microcontrollers and one remote device. The new script is `sdl_light_mixing_batch.py`. We tested keeping the same number of iterations as the first two parts (27) and keeping the same amount of time to run the experiment (81 iters, since we have three devices in parallel). Results are found in `results/combined/same_iter` and `results/combined/same_time`, respectively.

Usage: `python3 sdl_light_mixing_batch.py --pico_id <pico_id_1> <pico_id_2> ... --color <R> <G> <B> --num_iter <num_iter> --save_dir <save_dir>`

Example: `python3 sdl_light_mixing_batch.py --pico_id test e6616407e352322c --color 25 25 25 --num_iter 27 --save_dir results/combined/same_iter`
