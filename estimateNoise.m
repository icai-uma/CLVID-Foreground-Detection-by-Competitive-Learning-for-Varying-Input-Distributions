% Foreground Detection by Competitive Learning for Varying Input Distributions
% International Journal of Neural Systems 
% DOI: 10.1142/S0129065717500563  
% Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
% Date: July 2018 

% This function compute the noise of a sequence from the structure 
function noise = estimateNoise(model)

% The mean of the scene is used as the original frame
MuImage = double(shiftdim(squeeze(model.Mu(:,1,:,:)),1));

% The smoothing approach is applied
SmoothingFilter = fspecial('gaussian',[3 3],0.5);
SmoothFrame=imfilter(MuImage,SmoothingFilter);

% The difference between the two images is obtained
dif = (MuImage - SmoothFrame).^2;

% A 0.01-winsorized mean is applied instead of the standard mean because
% the first measure is more robust and certain extreme values are removed
dif2 = reshape(dif,size(dif,1)*size(dif,2),model.Dimension);
dif3 = sort(dif2);
idx = round(length(dif3)*0.99);
for NdxDim=1:model.Dimension
    dif3(idx:end,NdxDim) = dif3(idx-1,NdxDim);
end

noise = mean(dif3);

