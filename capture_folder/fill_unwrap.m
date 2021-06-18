clear
close all
%%
imgDir='./roudian_image/david/';    %总文件夹
usefolders = find_folders(imgDir);
len = length(usefolders);
for iii = 2:2
    nowdir = [imgDir usefolders{iii} '/'];
    files = dir([nowdir,'*.','mat']);
    matlen = length(files);
    for i = 1:matlen
        if files(i).name(1)=='u'
            load ([nowdir files(i).name])
            for col = 1:1024    % 先做列填充
                %  if files(i).name(end-4)=='1' || files(i).name(end-4)=='3'
                now_hl = unwrap(:,col);
                [new_now_hl] = my_fill(now_hl);
                unwrap(:,col) = new_now_hl;
            end
            for row = 1:1024    % 再做行填充
                %  else
                now_hl = unwrap(row,:);
                [new_now_hl] = my_fill(now_hl);
                unwrap(row,:) = new_now_hl;
                %  end
            end
            imagesc(unwrap)
            save(['unwrap' num2str(i)], 'unwrap')
        end
    end
end


%%
function [new_now_hl] = my_fill(now_hl)
no_nan = [];
for test_no_nan = 1:1024
    if isnan(now_hl(test_no_nan))
        continue;
    else
        no_nan = [no_nan test_no_nan];
    end
end
if length(no_nan)>100
    xt = no_nan(1):no_nan(end);
    new_now_hl = now_hl;
    pre_element = now_hl(xt(1));
    max_nan = 0;
    tmp_lead_id = 0;
    tmp_end_id = 0;
    for i = no_nan(2):no_nan(end)   % 这里的i代表总长度是1024中的那个位置
        if isnan(now_hl(i)) && ~isnan(pre_element)
            max_nan = max_nan+1;
            tmp_lead_id = i;
        elseif isnan(now_hl(i)) && isnan(pre_element)
            max_nan = max_nan+1;
        elseif ~isnan(now_hl(i)) && isnan(pre_element)
            tmp_end_id = i;
        end
        pre_element = now_hl(i);
        if tmp_end_id-tmp_lead_id>0 && max_nan>0 && max_nan<20
            [tmp_new_now_hl,~] = fillmissing(now_hl(tmp_lead_id-1:tmp_end_id),...
                'linear','SamplePoints',tmp_lead_id-1:tmp_end_id);
            max_nan = 0;
            new_now_hl(tmp_lead_id-1:tmp_end_id) = tmp_new_now_hl;
        end
    end
    
else
    new_now_hl = now_hl;
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

%%
function OutImg = Normalize(InImg)
ymax=255;ymin=0;
xmax = max(max(InImg)); %求得InImg中的最大值
xmin = min(min(InImg)); %求得InImg中的最小值
OutImg = round((ymax-ymin)*(InImg-xmin)/(xmax-xmin) + ymin); %归一化并取整
OutImg = uint8(OutImg);
end