% Foreground Detection by Competitive Learning for Varying Input Distributions
% International Journal of Neural Systems 
% DOI: 10.1142/S0129065717500563  
% Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
% Date: July 2018 

function []=cl_vid(params)
VideoFileSpec = params.VideoFileSpec;
deltaFrame = params.deltaFrame;
numFrames = params.numFrames;

% Create the structures of the stochastic approximation model
VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+1)));
FeatureFrame=VideoFrame/255;
model = createModel(FeatureFrame,params);

% Allocate scape for the set of images to initialise the model 
FirstFrames = zeros(size(FeatureFrame,1),size(FeatureFrame,2),size(FeatureFrame,3),model.NumPatterns);
FirstFrames(:,:,:,1) = FeatureFrame;

% Store the frames
for NdxFrame=2:model.NumPatterns
    VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+NdxFrame)));
    FeatureFrame=VideoFrame/255;
    FirstFrames(:,:,:,NdxFrame) = FeatureFrame;
end

% Initialize the model using a set of frames
MeanFirstFrames = median(FirstFrames,4);
model.Mu = repmat(MeanFirstFrames, [1 1 1 model.NumNeurons]);
model.Mu = model.Mu + 0.001 * randn (size(model.Mu));
model.Mu=shiftdim(model.Mu,2);

% Estimate the noise of the sequence
model.Noise = estimateNoise(model);

for NdxFrame=model.NumPatterns+1:numFrames
    VideoFrame=double(imread(sprintf(VideoFileSpec,deltaFrame+NdxFrame)));
    FeatureFrame=VideoFrame/255;
    [model,imMask,resp,imDistances]=updateBM_MEX(model,FeatureFrame);
	
	imMask = medfilt2(imMask, [5 5]);    
    imMask = double(imMask < 0.5);
    
    s = strel('disk',5);
    ID = imdilate(imMask,s);
    imMask = imerode(ID,s);
    
    % Fill holes (size 1) and remove objects with minimum area (10 pixels size)
    imMask = bwmorph(imMask,'majority');
    imMask = removeSpuriousObjects(imMask, 10);
    
    subplot(1,2,1),imshow(uint8(VideoFrame));
    subplot(1,2,2),imshow(imMask);
                                
    title(NdxFrame);
    pause(0.001);
end

