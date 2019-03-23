import numpy as np
from physics_sim import PhysicsSim



class Task(object):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(
        self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., action_size=4, 
        **kwargs):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = action_size


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    

class Target(Task): # original Udacity example implemented as subclass
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(
        self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):

        super(Target, self).__init__(
            init_pose=init_pose, 
            init_velocities=init_velocities, 
            init_angle_velocities=init_angle_velocities,)
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    
    
    
class Hover(Task):
    """
        Land Copter, i.e. reduce all velocities, and angular orientations to 0.  
        To avoid a situation where the copter crashes into the ground, require it to 
        hover at some designated low height as long as possible.  
        
        Presumably a subsequent control mechanism will take over to disengage the copter 
        and let it drop softly to the ground.
    """
    def __init__( 
        self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=10, hover_height=2, action_size=4, crash_height=0.1):
        
        # complete state size -- these parameters should be managed a lot better, but I just need
        # to get through this.
        super(Hover, self).__init__(
            init_pose=init_pose, 
            init_velocities=init_velocities, 
            init_angle_velocities=init_angle_velocities, 
            runtime=runtime, action_size=action_size)
        self.hover_height = hover_height
        self.state_size = self.action_repeat * 14
        self.crash_height = crash_height
        
    def step(self, rotor_speeds):
        # print("Stepping Hovor task with these speeds: {}".format(rotor_speeds))
        """Uses action to obtain next state, reward, done."""
        reward = 0
        substates = []
        
        if len(rotor_speeds)==1:
            rotor_speeds = np.repeat(rotor_speeds, 4)
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            substates.append(self.compile_state(self.sim))
        
        next_state = np.concatenate(substates)
        if next_state[2] < self.crash_height:
            print("I crashed!")
            done = True
        # print("Giving step state {}".format(next_state.shape))
        return next_state, reward, done
 
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.compile_state(self.sim)] * self.action_repeat)
        # print("Giving reset state {}".format(state.shape))
        return state
   
    def compile_state(self, sim):
        return np.concatenate([sim.pose, sim.v, sim.angular_v, [self.hover_height, self.sim.pose[2] - self.hover_height]])
        
    def get_reward(self):
        """We want neutral pose, <hover height> altitude, 0 velocities
        
            this is a soft landing function so being below our designated hover height is penalized
            more than staying above it
            
            a generalized hover might modify this height penalty function
        """
        height_error = self.sim.pose[2] - self.hover_height
        height_penalty = height_error**2 if height_error>0 else abs(height_error)**3
        return np.log(1/((
            abs(self.sim.pose[4:]).sum() +
            abs(self.sim.v).sum() +
            abs(self.sim.angular_v).sum() 
            ) * height_penalty ))
            