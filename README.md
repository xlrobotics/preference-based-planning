# preference-based-planning

## Docker Configuration

Pull docker image `docker pull abhibp1993/prefplanning:dev`

OR 

Build the docker image using Dockerfile in `preference-based-planning/docker`


## State Space Configuration

battery level from 0~12 with battery cap = 12 `(battery_range=13)`

uav coord `(4*4)`

cloud1.x `(4)`

cloud2.x `(4)`

since battery could only be `12` at the station, and `0-11` anywhere else, there are `2` stations in the gridworld

so we have the number of states outside the stations as `(uav.xrange*uav.yrange - 2)*cloud1.xrange*cloud2.xrange*(battery_range-1) = (4*4-2)*4*4*12 = 2688`

so we have the number of states at the stations as `2*cloud1.xrange*cloud2.xrange = 2*16=32`

then the total state space is `32+2688=2720` before the product

