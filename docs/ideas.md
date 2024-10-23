# Ideas for improvements

## Simulation Setup
- Configure a wider and larger environment (e.g. 700 m x 1300 m from [SAMS](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2600326) )
- Add small ocean current (e.g. 0.5 m/s with 45 degree wrt surge from [SAMS](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2600326))
- Faster vessel transit speed (e.g. 8 knots or ~4 m/s) with corresponding larger minimum turning radius.
- Higher concentration ice fields, i.e. close pack and very close pack ice (> 70 %)
  - See ice concentration definitions at [Ice Navigation in Canadian Waters](https://www.ccg-gcc.gc.ca/publications/icebreaking-deglacage/ice-navigation-glaces/page07-eng.html)
- Classify ship-ice contacts into collisions and resting contacts and compute the impact force accordingly (see Section 3.1 on Ship and Floe collision model in [this paper](https://www.sciencedirect.com/science/article/pii/S095183392030160X5))

## Code
- Use ROS to communicate between the planner and the simulator (can possibly also use this new ROS interface for NRC experiments)
- Faster rendering of the simulation than the current pyplot animation
- Use GPU acceleration to compute swath costs
- A parent/abstract class to remove code duplication in planner code **/planners**
- Additional baselines:
  - Constant collision cost function (i.e. every grid cell occupied by ice has a constant cost value)
- Replace skimage.draw.polygon with Pillow ImageDraw.polygon (it might be a lot faster)

## Planning
- Do motion planning using the dynamics of the vessel
- Make speed variable during planning
- Use predictions of ice motion in planner
- Set a constraint for the maximum impact force exerted on the ship (see Figure 1 in [this paper](https://arxiv.org/pdf/2209.02389))
- Account for the multi-modal nature of the costmap which can result in several good candidate paths that have very similar costs
  but are very different in terms of the path taken
- Account for noise in ice data (see ice segmentation in sample videos from NRC experiments)
- Cost function improvements:
  - Different cost function depending on the size of the ice floe (see Section 2.1 on 'Energy Conservation' in [this paper](https://www.mdpi.com/2077-1312/10/2/165))
    or the type of ice (see Section 1.1.1 Sea-ice types in [Ice Navigation in Canadian Waters](https://www.ccg-gcc.gc.ca/publications/icebreaking-deglacage/ice-navigation-glaces/page07-eng.html))
  - Different cost function depending on whether the ship is colliding or pushing ice 
  - Include ice velocity as a variable in the cost function
  - Learn the cost function from data 
  - Feedback mechanism to update cost function parameters based on ice conditions measured online

## Control
- Use Fossen's [nonlinear supply model](https://github.com/cybergalactic/MSS/blob/master/VESSELS/SIMosv.m) to replace the linear dynamics model in supply.py.
  This script also includes code for an improved controller.