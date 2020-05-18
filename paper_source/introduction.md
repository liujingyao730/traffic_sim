# Introduction

- With the urbanization process, the importance of large-scale traffic simulation has become increasingly prominent. and macroscopic traffic model is curcial to large-scale traffic simulation. Cell transmission model proposed by Daganzo is one of the most well-konwn macroscopic traffic model, which has alrealy gotten lots of practical application
- Traditional CTM is a numerial method to solve differential equations of LWR model. In traditional CTM model, hand design piecewise functions are used to cucaulate the flow or density transfer from current cell to next cell, which could be formalized as:

$$
\begin{aligned}
    y(t,s) &= f(state(s), state(s+1), state(s-1), ...) \tag{1}\\
    n(t,s) &= n(t-1,s) - y(t,s) + y(t,s-1)
\end{aligned}
$$
- where $y(t, s)$ repesent the flow or density transfer from cell $s$ to cell $s+1$ at time slot $t$, $n(t, s)$ repesent the number or density of cell $s$ at time slot $t$, and $state(s)$ repesent some state of cell $s$, like density, traffic flow, occupancy, etc. $f(\cdot)$ repesents a traffic estimate fucntion (usually hand-desgined), which is a key component of CTM.Both traffic state and traffic state estimate fucntions vary with different models.
- Based on such a framework, researchers have introduced several extension to the traditional CTM to enhance its applicability and accuracy. For instance, Alecsandru(2006), Boel and Mihaylova (2006), Zhong and Sumalee (2008), Sumalee et al. (2011) proposed stochastic CTM, which can handle errors due to random fluctuations of parameters in the fundamental diagram by introducing quantified randomness on top of the original CTM; Work et al, 2008 proposed the velocity-CTM, which could estimates the traffic density over time from the acquired velocity data based on the original CTM; The study by Laval and Daganzo, 2006; Carey et al., 2015 . introduced the behavior of lane changing; the study by Gomes and Horowitz (2006), Gomes et al. (2008). explored how CTM should be applied in a road network with on- and off-ramps
- Some researchers try to reduce distortion during simulation by introducting a reasonable description of traffic heterogeneities. To produce platoon dispersion phenomena, LWR was extended to multiclass LWR (Wong and Wong, 2002). The model introduces the notion of classes defined heterogeneous driving on a freeway. The model divides vehicles into passenger vehicles(PV), which have higher speeds and occupy less space, and heavy vehicle(HV), which have slower speeds and require more space.Researchers have developed a number of multi-class CTM models for simulating traffic flows with a certain percentage of HV on freeways (goduy and Liu, 2007; Van Lint et al.,2008; Ngoduy, 2011; Szeto et al., 2011; Mesa-Arango and Ukkusuri, 2014; Liu et al., 2015; Qian et al., 2017; Zhan and Ukkusuri, 2017). n order to produce platoon dispersion in a general topology, CTM was extended to multiclass CTM by Tuerprasert and Aswakul (2010), denote as M-CTM. Based on the reported results, M-CTM is found be able to produce platoon dispersion well without compromising on the model’s computational complexity. Kamonthep Tiaprasert et al (2017) proposed model FM-CTM, standing for the multiclass cell transmission model with FIFO property, which enhanced the accuracy of multiclass CTM model.
- but the traditional CTM like M-CTM and FM-CTM still facing several issues. In traditional CTM, hand-desgined traffic state estimate fucntion could can roughly simulate some features of the traffic flow, but because the function designed by hand does not fully capture the complex nonlinear features of the traffic flow, it often leads to error accumulation on long sections of the road; furthermore, in order to avoid excessive error accumulation, many models will limit the expansion of the error by manually setting conditions, which creates local distortions
- to address those issue, we develope a attention recurrent neural network based CTM
- Recurrent neural network has been proved very powerful in Processing time-series data and achieved great success on many areas such as speech recognition, Natural language processing, trajectory prediction, Traffic flow prediction etc.LSTM is one of the most well-known recurrent neural network, which could formalized as :
$$
\begin{aligned}
    f_t &= \sigma_g(W_fx_t + U_fh_{t-1}+b_f) \\
    i_t &= \sigma_g(W_ix_t + U_ih_{t-1}+b_i) \\
    o_t &= \sigma_g(W_ox_t + U_oh_{t-1}+b_o) \\
    c_t &= f_t \circ c_{t-1} + i_t \circ \sigma_c(W_cx_t+U_ch_{t-1}+b_c) \\
    h_t &= o_t \circ \sigma_h(c_t) \\
\end{aligned}
$$
- researchers introduced several variant based on aforementioned LSTM model. different structs and mechanisms are developed to handle different problem
- traditional CTM framework proved to be powerful in traffic simulation, but the hand designed function in CTM is the bottle-neck of more accurate and realistic simulation
- recurrent neural network based model could learn essential rule of traffic from data. Vanilla recurrent neural network such as LSTM is not suitable for spatial temporal problem we are facing now,  but with some necessary and targeted change, we can use it as a key component in our model.
- For traffic simulation, we need to  handle both temporal information and spatial information.  so we use Vanilla LSTM model to capture temporal information in each cell, and intergate a special component to capture the influence of other cells bring to the current cell.
- our contribution is
  - we design a Attention base LSTM Cell transmission model (AL-CTM) which could be significantly accurate than traditional CTM on both cell level and road segment level
  - our model is trained on data which collected from microscopic model simulation, that make our simulation more realistic than tradition CTM model