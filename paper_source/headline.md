## introduction
- macroscopic traffic model is curcial to large scale traffic simulation
- Cell transmission model proposed by Daganzo is one of the most well-konwn macroscopic traffic model, which has alrealy gotten lots of practical application
#### traditional CTM
- In traditional CTM model, hand design piecewise functions are used to cucaulate the flow or density transfer from current cell to next cell, which could be formalized as:
- $$y(t,s) = f(state(s), state(s+1), state(s-1))$$
- $$n(t,s) = n(t-1,s) - y(t,s) + y(t,s-1)$$
- where $y(t, s)$ repesent  the flow or density transfer from cell s to cell s+1 at time slot t, $n(t, s)$ repesent the number or density of cell s at time slot t, and $state(s)$ repesent some state of cell s, lisk density, traffic flow, tracffic condition...it varies with different models
- traditional CTM is a numerial method to solve differential equations of LWR model
- researchers have introduced several extension to the traditional CTM to enhance its applicability and accuracy
- prior work of M-CTM and FM-TCM have shown its ability on simulate mulitclass traffic
- but the traditional CTM like M-CTM and FM-CTM still facing several issues:
  -  hand-designed function in CTM may cause error accumulation problem in long road
  -  most of CTM can only work on specific assumptions, which may limit simluation
- to address those issue, we develope a attention recurrent neural network based CTM
#### neural network
- recently neural network method have achieve great success on many area
- Recurrent neural network has been proved very powerful in Processing time-series data
- LSTM is one of the most well-known recurrent neural network, which could formalized as :
- $$f_t = \sigma_g(W_fx_t + U_fh_{t-1}+b_f)$$
- $$i_t = \sigma_g(W_ix_t + U_ih_{t-1}+b_i)$$
- $$o_t = \sigma_g(W_ox_t + U_oh_{t-1}+b_o)$$
- $$c_t = f_t \circ c_{t-1} + i_t \circ \sigma_c(W_cx_t+U_ch_{t-1}+b_c)$$
- $$h_t = o_t \circ \sigma_h(c_t)$$
![lstm](LSTM.png)
- researchers introduced several variant based on aforementioned LSTM model. different structs and mechanisms are developed to handle different problem
#### our contribution
- traditional CTM framework proved to be powerful in traffic simulation, but the hand designed function in CTM is the bottle-neck of more accurate and realistic simulation
- recurrent neural network based model could learn essential rule of traffic from data. Vanilla recurrent neural network such as LSTM is not suitable for spatial temporal problem we are facing now,  but with some necessary and targeted change, we can use it as a key component in our model.
- For traffic simulation, we need to  handle both temporal information and spatial information.  so we use Vanilla LSTM model to capture temporal information in each cell, and intergate a special component to capture the influence of other cells bring to the current cell.
- our contribution is
  - we design a Attention base LSTM Cell transmission model (AL-CTM) which could be significantly accurate than traditional CTM on both cell level and road segment level
  - our model is trained on data which collected from microscopic model simulation, that make our simulation more realistic than tradition CTM model

## our model
#### framework
- first of all, we need to define the problem we met:
- $$x^s_t = [o_t^s, i_t^s, N_t^s]^T$$
- $$X^s = [x_1^s,x_2^s,...,x_{obs}^s]$$
- $$Y^s = [o_{obs+1}^s,...,o_{end}^s]^T$$
- $$X=[X^1, X^2,...X^n] \stackrel{f}{\longrightarrow} Y=[Y^1, Y^2, ..., Y^n]$$
- $o_t^s, i_t^s$ respesents the number of vehicle come out and into cell s during time slot t, $N^s_t$ repesent the number of vehicles in cell s at time t, all those combine together make the input vector of cell s at time t
- traditional CTM use a hand design fucntion, In our model, we keep the framework of traditional CTM but replace the hand design function  $f$ with a neural network $\mathcal{M}$
#### detail
- hidden state and cell state in LSTM is the temporal feature which captured from input data, so we desgin a component which could receive and process the information from other cells:
- $$h_{sp} = ATTENTION(...,h_{t-1}^{s-2}, h_{t-1}^{s-1}, h_{t-1}^{s+1}, h_{t-1}^{s+2},...)$$
- $$\hat{h}_{t-1}^s = h_{t-1}^s \circ \sigma_g(W_{sf}h_{sp} + b_{sf}) $$
- $$\hat{c}_{t-1}^s = c_{t-1}^s \circ \sigma_g(W_{si}h_{sp} + b_{si}) $$
- $$h^s_t, c_t^s = LSTM(\hat{h}_{t-1}^s,\hat{c}_{t-1}^s, x_t^s)$$
![frame](frame.png)
- then the inputs of next time slot could caculate as:
- $$o_{t+1}^s = MLP(h_t^s), i_{t+1}^s=o_{t+1}^{s-1}, N^s_{t+1}=N^s_t+i_{t+1}^s-o_{t+1}^s$$
#### attention mechanism
- attention mechanism have become almost a de facto standard in many sequence-based tasks. in our model, we use attention mechanism to handle spatial information:
- $$e_{ij} = LeakyReLU(W_a[h_i, h_j])$$
- $$\alpha_{ij} = softmax(e_{ij})=\frac{\exp(e_{ij})}{\sum_{j\in\mathcal{N_i}\exp(e_{ij})}}$$
- $$h_i' = \sigma(\sum_{j\in\mathcal{N}_i}\alpha_{ij}h_j)$$
- $h_i'$ repesents the result of one head in our multi-head attention layer, concatenate all the results from each head, we get the spatial  information:
- $$h_{sp}=CONCAT(h_i', h_i'',...)$$
- for those cells at the end or beginning of a road, which don't have upstream or downstream cells in simulation, we use learnable parameters to act as the spatial information they need
![param](param.png)
$h_{spatial}^s = [h_{t-1}^{s-2}, h_{t-1}^{s-1}, h_{t-1}^{s+1}, h_{t-1}^{s+2}]$
$h_{spatial}^0 = [p_{before}^1, p_{before}^0, h_{t-1}^{1}, h_{t-1}^{2}]$
$h_{spatial}^1 = [p_{before}^0, h_{t-1}^0, h_{t-1}^{2}, h_{t-1}^{3}]$
$h_{spatial}^{end} = [h_{t-1}^{end-2}, h_{t-1}^{end-1},p_{after}^0, p_{after}^1]$
$h_{spatial}^{end-1} = [h_{t-1}^{end-3}, h_{t-1}^{end-2},h_{t-1}^{end}, p_{after}^1]$
- the loss function we designed requires the model to be more accurate not only on the output flow of a cell, but also the number of vehicles in it
- $$L=\alpha Loss(o_t, \hat{o}_t)+\beta Loss(N_t, \hat{N}_t)$$

## Numerical experiments and results
#### dataset description
- the Synthetic data we used is generated by a microscopic model based traffic simulation software SUMO
- the network of simlution is a 1.5km long straight roadway with 6 lanes. In this road, traffic speeds are restricted to 50km/h, There are two classes of vehicles: PVs and HOVs. both of the two classes vehicles follow Gauss distribution, the average  speed of PVs and HOVs are 45km/h and 35km/h, and the variances are 5km/h and 2.5km/h respectively. Time interval is 5 s. the road is divided equally into a series of 30 cascading cells with 50 meters in length. Total simulation time is 3600 s in every cases.
-  there are totally 18 simulation cases, with 6 different mixing ratios of HOVs and PVs range from 0.05 to 0.3,and 3 different boundary input flow scenes. shown as follow:
-  number of vehicle in the road of three different input flow scene
-  scene 1
![base](base_flow.png)
- scene 2
![test](test_flow.png)
- scene 3
![change](change_flow.png) 
- the real traffic data was obtained from Next Generation Simulation (NGSIM) project The site is a segment of Interstate 80 in Emeryville (San Francisco), collected between 4:00 p.m. and 4:15 p.m., 5:00 p.m. and 5:15 p.m., and 5:15 p.m. and 5:30 p.m. The segment is about 400 m and with 6 lanes.The overall percentage of HVs is 5%. However, only the vehicle trajectory data between 4:00 p.m. and 4:15 p.m. were reconstructed over 15 min (Punzo et al., 2011; Montanino and Punzo, 2015). 
- Time interval is 5s with 132 total time slots. The densities of two vehicle classes, PVs and HVs, of each cell along with the boundary conditions are obtained from the vehicle trajectory data. The length of a PV, from the data, is 4.3 m. On the other hand, the length of an HV varies between 6.4 and 23 m. Here, we use the average length, 14 m, as HVs’ length.
#### training detail
- we use data in scene 1 and scene 2 to train our model and calibrate parameters of FM-CTM, and scene 3 to test the preformance of two different mdoel, we train our model on the data 50 epochs with learning rate as 0.005. the size of hidden state is 64 and number of heads as 4.
-  we integrate "scheduled sampling" into the model, where we feed the model with either the ground truth observation with probability $\epsilon$ or the prediction by the model with probability $1 − \epsilon$ at the ith iteration. During the training process, $\epsilon$ gradually decreases to 0 to allow the model to learn the testing distribution, in our training process $\epsilon$ started with 1 and shrink 0.1 every epoch until 0. 
#### experiment result
- in our experiment, we not only test and compare our model and FM-CTM on each single cell, but also try to figure out the performance of different model on the whole road segment
- before comparing our model with other traditional CTM, we make some ablation experiments to show that our model struct do make sense
- the result shows that, attention mechanism could provide more nonlinearity, which will make the model more accurate on both cell level and segment level
![ablation](ablation.png)
- Learnable parameters could improve the result on the begining and end cells, and further improver the accuracy on the whole segment
- we compare our model with CTM on both Synthetic data and real data
- For synthetic data, microscopic simulate software SUMO is adopted as ground truth
- the real traffic data was obtained from Next Generation Simulation (NGSIM) project
- on the data generated by SUMO, our model shows more accuracy on both cell level and segment level
![SUMO_result](SUMO_result.png)
![flow](SUMO_flow_comp.png)
- because of hand-designed function in traditional CTM there will be some realistic flow in spatial-temproal heat map, which do not appear in our model result
![heat](heat.png)
- our model trained on synthetic data could also achieve better preformance on both cell level and segment level
![real_Data](real_data.png)