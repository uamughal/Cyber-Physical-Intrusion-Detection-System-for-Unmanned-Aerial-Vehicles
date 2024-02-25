# Cyber-Physical Intrusion Detection System for Unmanned Aerial Vehicles

The increasing reliance on unmanned aerial vehicles
(UAVs) has escalated the associated cyber risks. While machine
learning has enabled intrusion detection systems (IDSs), current
IDSs do not incorporate cyber-physical UAV features, which
limits their detection performance. Additionally, the lack of
public UAV’s cyber and physical datasets to develop IDS hinders
further research. Therefore, this paper proposes a novel IDS
fusing UAV cyber and physical features to improve detection
capabilities. First, we developed a testbed that includes UAV,
controller, and data collection tools to execute cyber-attacks
and gather cyber and physical data under normal and attack
conditions. We made this dataset publicly available. The dataset
covers a range of cyber-attacks including denial-of-service,
replay, evil twin, and false data injection attacks. Then, machine
learning-based IDSs fusing cyber and physical features were
trained to detect cyber-attacks using support vector machines,
feedforward neural networks, recurrent neural networks with
long short-term memory cells, and convolutional neural networks.
Extensive experiments were conducted on varying complexity
and range of attack training data to explore whether (a) fusion
of cyber and physical features enhances detection performance
compared to cyber or physical features alone, (b) fusion enhances
detection when IDS is trained on a single attack type and tested
on unseen attacks of varying complexity, (c) fusion enhances
performance when the range of attack training data increases and
models are tested on unseen attacks. Answering these research
questions provides insights into IDS capabilities using cyber,
physical, and cyber-physical features under different conditions.


# Research Questions:
We investigated the following research questions in this paper:
- Will the fusion of cyber and physical features improve
detection compared to cyber or physical features alone?
- Will the complexity level of the attack training data
impact the ability of the IDS to detect unseen attacks
with various complexities?
- Will the range of the attacks included in the training data
impact the ability of the IDS to detect unseen attacks with
various complexities?
