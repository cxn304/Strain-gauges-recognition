clear
close all
%% read files
imgDir='./16224489993/';    %×ÜÎÄ¼þ¼Ð
files = dir([imgDir,'*.','png']);
mlen = length(files);
for j = 1:mlen
    image_name = [imgDir files(j).name];
    imgs = imread(image_name);       
    new_imgs = imgs(:,129:1152);
    imwrite(new_imgs,image_name);
end