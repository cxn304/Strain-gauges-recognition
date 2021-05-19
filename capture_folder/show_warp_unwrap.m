clear
close all
%% read files
imgDir='./roudian_image/0/';    %总文件夹
usefolders = find_folders(imgDir);
len = length(usefolders);
for iii = 1:1
    nowdir = [imgDir usefolders{iii} '/'];
    files = dir([nowdir,'*.','mat']);
    mlen = length(files);
    for dMoveUpDown = 10:10:200
        for dMoveLeftRight = 10:10:200
            figure
            set(gcf,'position',[0,50,900,400]);
            for j = 1:mlen
                image_name = [imgDir usefolders{iii} '/' files(j).name];
                tmp_name = files(j).name;
                tmp_name = tmp_name(1:end-4);
                image_now = load(image_name);
                if files(j).name(1)=='u'
                    image_now = image_now.unwrap;
                    saveImageDir = ['./train_unwrap/' tmp_name '_' ...
                    num2str(dMoveUpDown) '_' num2str(dMoveLeftRight) '.mat'];
                else
                    image_now = image_now.wrapped;
                    saveImageDir = ['./train_wrapped/' tmp_name '_' ...
                    num2str(dMoveUpDown) '_' num2str(dMoveLeftRight) '.mat'];
                end
                image_new = zeros(size(image_now));
                image_new(1:end-dMoveUpDown+1,1:end-dMoveLeftRight+1)...
                    = image_now(dMoveUpDown:end,dMoveLeftRight:end);
                image_new(isnan(image_new))=0;
                save(saveImageDir,'image_new');
                subplot(2,4,j)
                imagesc(image_new)
            end
            close all
        end
    end
end
%%
function show_images(imgDir,usefolders,len)
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