clear
%% read files
imgDir='./roudian_image/0/';    %总文件夹
image_folders = find_folders(imgDir);
len = length(image_folders);
for i = 1:len
    imgs = find_folders([imgDir image_folders{i}]);
    for j = 1:length(imgs)
        full_img_name = strsplit(imgs{j},'.');
        if full_img_name{end} == 'png'
            tmpimg = imread([imgDir image_folders{i} '/' imgs{j}]);
            [mm,nn] = size(tmpimg);
            if nn == 1280
                tmp_img = tmpimg(:,129:end-128);
                imwrite(tmp_img,[imgDir image_folders{i} '/' imgs{j}]);
            end
        end
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