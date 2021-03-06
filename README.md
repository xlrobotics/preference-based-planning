# preference-based-planning

## Docker Configuration

Pull docker image `docker pull abhibp1993/prefplanning:dev`

OR 

Build the docker image using Dockerfile in `preference-based-planning/docker`

## Running Instruction
1. run `initialize_mdp_gridworld.py` to generate the product mdp, AS region and vector value function. The result will be stored as an MDP object in the pickle file `prod_MDP_gridworld_v2.pkl`.
2. run `calc_result_vector_V.py` to generate the spi policy and store as a dictionary in the pickle file `PI_prod_MDP_gridworld_v2.pkl`.
3. run `gridworld.py` to test and visualize the policy via pygame env. The configuration of the env is defined in `toy_example.yaml`.


## State Space Configuration （cloud without staying option）

battery level from 0~12 with battery cap = 12 `(battery_range=13)`

uav coord `(4*4)`

cloud1.x `(4)`

cloud2.x `(4)`

since battery could only be `12` at the station, and `0-11` anywhere else, there are `2` stations in the gridworld

so we have the number of states outside the stations as `(uav.xrange*uav.yrange - 2)*cloud1.xrange*cloud2.xrange*(battery_range-1) = (4*4-2)*4*4*12 = 2688`

so we have the number of states at the stations as `2*cloud1.xrange*cloud2.xrange = 2*16=32`

then the total state space is `32+2688=2720` before the product


## State Space Configuration （cloud with staying option）

battery level from 0~12 with battery cap = 12 `(battery_range=13)`

uav coord `(4*4)`

cloud1.x `(4)`

cloud2.x `(4)`

stay_counter1 '(3: 0, 1, 2)'

stay_counter2 '(3: 0, 1, 2)'

since battery could only be `12` at the station, and `0-11` anywhere else, there are `2` stations in the gridworld, the counter of the clouds are always working no matter the uav is trapped or not,

so we have the number of states outside the stations as `(uav.xrange*uav.yrange - 2)*cloud1.xrange*cloud2.xrange*(battery_range-1)*counter1_range*counter2_range = (4*4-2)*4*4*12*3*3 = 2688 * 9 = 24192` 

so we have the number of states at the stations as `2*cloud1.xrange*cloud2.xrange = 2*16*9=32*9 = 288`

then the total state space is `288+24192=24480` before the product
