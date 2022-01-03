# navstack-gym
Simulation environment of task with autonomous mobile robot using Navigation Stack.


<img src='https://user-images.githubusercontent.com/41321650/145752733-cb9f80d7-2647-464f-8ee6-57ee3e53dbf2.png' width=100%>

In this environment, the agent do action of instructing relative navigation goal pose and observe a subjective occupancy map.

<img src='https://user-images.githubusercontent.com/41321650/145782630-5cda4862-948c-4995-9739-ca002a77ae68.GIF' width=100%>


## Implemented Task
`TreasureChestRoom` :   
Agent aim to open chests in unknown rooms with keys and discover as much treasure as possible.

The rooms in which agent spawned are randomly generated such as the following structure.

Yellow cube is key, and cyan cube is treasure chest. Each object will be generated based on different set of placing rules.

<img src='https://user-images.githubusercontent.com/41321650/144612303-07df02c0-b4af-46e0-8eea-36d905246f76.png' width=100%>


## Installation

```
pip install navstack-gym
```

## Usage

I'll add the note later.

example:

```python
import gym
import navstack_gym

env = gym.make('VisibleTreasureHunt-v0')
obs = env.reset(is_generate_pose=True, is_generate_room=True, obstacle_count=10)

imgs = []
imgs.append(env.render('rgb_array'))

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    imgs.append(env.render('rgb_array'))
```

<img src='https://user-images.githubusercontent.com/41321650/147936159-d6691e8e-8216-465b-a4a6-bd8748a3b010.gif' width=100%>
