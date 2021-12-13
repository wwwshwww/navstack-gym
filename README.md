# navstack-gym
Environment of task with Autonomous mobile robot using Navigation Stack.


<img src='https://user-images.githubusercontent.com/41321650/145752733-cb9f80d7-2647-464f-8ee6-57ee3e53dbf2.png' width=100%>

In this environment, the agent do action of instructing relative navigation goal pose and observe subjective occupancy map.

<img src='https://user-images.githubusercontent.com/41321650/145752731-b2176aaa-7b3c-450a-9da6-e26ddae0a5bb.gif' width=100%>


## Implemented Task
`TreasureChestRoom` :   
Agent aim to open chests in unknown rooms with keys and discover as much treasure as possible.

The rooms in which the agents are spawned are randomly generated such as the following structure.

<img src='https://user-images.githubusercontent.com/41321650/144612303-07df02c0-b4af-46e0-8eea-36d905246f76.png' width=100%>


## Installation

`pip install navstack-gym`

## Usage

I'll add a note later.

```python
import gym
import navstack_gym

env = gym.make('TreasureChestRoom-v0')
obs = env.reset(is_generate_pose=True, is_generate_room=True)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```
