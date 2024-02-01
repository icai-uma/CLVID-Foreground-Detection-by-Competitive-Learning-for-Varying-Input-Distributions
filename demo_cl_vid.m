% Foreground Detection by Competitive Learning for Varying Input Distributions
% International Journal of Neural Systems 
% DOI: 10.1142/S0129065717500563  
% Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
% Date: July 2018 

% Demo code for the paper
tic
clear all

rng('default'); % the seed of randomness always is the same
                % so we always get the same results

% Video to be processed (the selected video is part of the highway sequence, taken from
% changedetection.net)
VideoFileSpec{1}='baseline/highway';
deltaFrame{1}=0;
numFrames{1}=400;

NdxVideo=1;

params.VideoFileSpec = 'video/input/in%06d.jpg';
params.deltaFrame = deltaFrame{NdxVideo};
params.numFrames = numFrames{NdxVideo};
params.videoSelected = VideoFileSpec{NdxVideo};

params.NumNeurons = 6;
params.CoefWinner = 0.05;
params.CoefAll = 0.0001;
params.DistanceThreshold = 0.01;

cl_vid(params);


toc
