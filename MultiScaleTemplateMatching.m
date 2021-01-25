%%
%Require:mexopencv3.0
%多尺度模板匹配
%原理：根据原始模板图像预先产生一系列不同尺度的模板
%检测的时候分别用上述各个尺度的模板遍历图像

%%
clear;
clc;

srcImg_rgb=imread('0132.jpg');
tImg=imread('hh.jpg');
srcImg=rgb2gray(srcImg_rgb);

[tImg_rows,tImg_cols]=size(tImg);
sprintf('原始模板的尺度(行，列)为:[%d,%d]',tImg_rows,tImg_cols)

%总共10种尺度
scale_step=1.1;
%一系列不同尺度
size_multi=zeros(5,2);
size_multi(1,:)=[tImg_rows,tImg_cols]; %原始尺度
%smaller;
for i=1:5
    size_multi(i+1,:)=[tImg_rows,tImg_cols]/(scale_step^i);
end
%larger
for i=1:5
    size_multi(i+5,:)=[tImg_rows,tImg_cols]*(scale_step^i);
end

%尺度转为整数
size_multi=uint16(size_multi);

%多尺度匹配
resMap=cell(10,1);
peak=zeros(10,1);
position=cell(10,1);
figure('name','多尺度模板');
for i=1:10
    %模板尺度变换
    tempImg=imresize(tImg,size_multi(i,:));
    %显示模板
    subplot(2,5,i);imshow(tempImg);
    %相关匹配
    resMap{i}=cv.matchTemplate(srcImg,tempImg,'Method','CCorrNormed');
    %定位峰值
    peak(i)=max(max(resMap{i}));
    %峰值对应的行，列
    [row,col]=find(resMap{i}==peak(i));
    position{i}=[row,col];
end


%显示各个尺度的匹配结果
%figure('name','matchResult');
for i=1:10
    %绘制矩形
    showImg=cv.rectangle(srcImg_rgb,[position{i}(2),position{i}(1),size_multi(i,2),size_multi(i,1)],'Color',[255,255,0]);
    figure(i+1)
    imshow(showImg)
    sprintf('尺度%d的匹配峰值为:%f',i,peak(i))
    %subplot(2,5,i);subimage(showImg);
end

%定位最大峰值
maxPeak=max(peak);
maxIndex=find(peak==maxPeak);
sprintf('最大峰值为:%d',maxPeak)
sprintf('最大峰值对应的模板尺度(最佳尺度)为:[%d,%d]',size_multi(maxIndex,1),size_multi(maxIndex,2))
%显示结果
resultImg=cv.rectangle(srcImg_rgb,[position{maxIndex}(2),position{maxIndex}(1),size_multi(maxIndex,2),size_multi(maxIndex,1)],'Color',[255,255,0]);
figure('name','最佳匹配');
imshow(resultImg);

