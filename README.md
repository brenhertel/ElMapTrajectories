# ElMapTrajectories

Implementation of Elastic Maps for trajectory reproduction use.

Corresponding paper can be found for free [here](https://arxiv.org/abs/2208.02207), please read for method details. Accompanying video available [here](https://youtu.be/rZgN9Pkw0tg).

Learning from Demonstration (LfD) is a popular method of reproducing and generalizing robot skills from given human demonstrations. In this paper, we propose a novel optimization-based LfD method that encodes demonstrations as elastic maps. An elastic map is a graph of nodes connected through a mesh of springs. The formulated optimization problem in our approach includes three quadratic objectives with natural and physical interpretations. The main term rewards the mean square error in the Cartesian coordinate. The second term penalizes the non-equidistant distribution of points resulting in the optimum total length of the trajectory. The third term rewards smoothness while penalizing nonlinearity. Additionally, our proposed LfD representation forms a convex problem that can be solved efficiently with local optimizers. We examine methods for constructing and weighting the elastic maps and study their performance in robotic tasks. We also evaluate the proposed method in several simulated and real-world experiments using a UR5e manipulator arm, and compare it to other LfD approaches to demonstrate its benefits and flexibility across a variety of metrics.

<img src="https://github.com/brenhertel/ElMapTrajectories/blob/main/pictures/paper_figures/pressing_reproduction.png" alt="" width="300"/> <img src="https://github.com/brenhertel/ElMapTrajectories/blob/main/pictures/paper_figures/robot_pressing.png" alt="" width="418"/>

This repository implements the method described in the paper above using Python. Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2022ElMap,
  title={Robot Learning from Demonstration Using Elastic Maps},
  author={Hertel, Brendan and Pelland, Matthew and S. Reza Ahmadzadeh},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
  organization={IEEE}
}
```
