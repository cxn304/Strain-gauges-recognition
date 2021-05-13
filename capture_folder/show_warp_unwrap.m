clear
close all
%% read files
imgDir='./roudian_image/0/';    %总文件夹
usefolders = find_folders(imgDir);
len = length(usefolders);
for iii = 1:len
    nowdir = [imgDir usefolders{iii} '/'];
    files = dir([nowdir,'*.','mat']);
    displacement = dir([nowdir,'*.','txt']);
    mlen = length(files);
    figure
    set(gcf,'position',[0,50,1800,400]);
    for j = 1:mlen
        image_name = [imgDir usefolders{iii} '/' files(j).name];
        image_now = load(image_name);
        if files(j).name(1)=='u'
            image_now = image_now.unwrap;
        else
            image_now = image_now.wrapped;
        end
        subplot(2,4,j)
        imagesc(image_now)
        colorbar
        title([files(j).name ' ' displacement(1).name])
    end
    close all
end

%%
function [usefolders] = find_folders(dirs)
% 解包folder
folders = dir(dirs);
usefolders = {};
k=1;
for i = 1:length(folders)
    if ~isempty(strfind(folders(i).name,'16'))
        usefolders{k}=folders(i).name;
        k = k+1;
    end
end
end