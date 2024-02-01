% Foreground Detection by Competitive Learning for Varying Input Distributions
% International Journal of Neural Systems 
% DOI: 10.1142/S0129065717500563  
% Authors: Ezequiel López-Rubio, Miguel A. Molina-Cabello, Rafael M. Luque-Baena, Enrique Domínguez
% Date: July 2018 

% The objects in the image 'bw' with few pixels (less than 'min_area') will
% be removed in the output image. 
function bw_out = removeSpuriousObjects(bw, min_area)

[L,num_blobs] = bwlabel(bw);
        
bw_out = zeros(size(bw,1),size(bw,2));
for i=1:num_blobs 
    Ob = (L == i);
    area = bwarea(Ob);
    if area > min_area
        bw_out = bw_out + Ob;
    end
end
