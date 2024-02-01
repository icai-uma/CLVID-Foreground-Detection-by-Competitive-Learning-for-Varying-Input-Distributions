% Foreground Detection by Competitive Learning for Varying Input Distributions
% International Journal of Neural Systems 
% DOI: 10.1142/S0129065717500563  
% Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
% Date: July 2018 

% The model structure is built from a frame of the analysed sequence
function Model = createModel(frame,params)

NumImageRows = size(frame,1);
NumImageColumns = size(frame,2);
Dimension = size(frame,3);

% Epsilon is the step size which regulates how quick the learning process is
% Valid values are shown in the paper 
Model.Epsilon = 0.01; 

Model.DistanceThreshold = params.DistanceThreshold; % Threshold distance between the winning neuron and the data to determine if it is background or foreground 
Model.NumNeurons = params.NumNeurons; % Number of neurons 6 12 18
Model.CoefWinner = params.CoefWinner; % Winning neuron update coefficient 0.20 0.30 0.50
Model.CoefAll = params.CoefAll; % Neurons update coefficient. Note that CoefAll is much smaller than CoefWinner, and CoefAll+CoefWinner<1  0.01 0.05 0.1

Model.NumPatterns = 100; % Number of used patterns to initilise the model
Model.H = 2; % h is a global smoothing parameter to compute the noise (by default is 2)
Model.NumCompGauss=1; % Number of Gaussian distributions (it properly works with 1)
Model.NumCompUnif=1; % Number of uniform distributions (it properly works with 1)
Model.Z = 1000; % Maximum number of consecutive frames in which a pixel belongs to the foreground class 
               % It is assumed that it is computed offline by analising 
               % a subset of frames of the sequence (by default 250)
Model.CurrentFrame = 1; % Indicates the current frame (at the begining 1)
Model.KernelProcesses = 4; % Number of CPU kernels to parallel the process
Model.Dimension=Dimension; % Number of features of each pixel

Model.NumComp=Model.NumCompGauss+Model.NumCompUnif; % Total number of distributions
Model.Log = 'temp.txt'; % Name of the log file

% Allocating space for work variables
Model.Mu=zeros(Dimension,Model.NumNeurons,NumImageRows,NumImageColumns);
Model.Counter=zeros(NumImageRows,NumImageColumns);



