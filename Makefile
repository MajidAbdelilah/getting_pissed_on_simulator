all:
	icpx -fsycl -g -xhost -Ofast  -Dicpx main.cpp my_random.cpp -L./lib -l:libraylib.a -o getting_pissed_on_simulator