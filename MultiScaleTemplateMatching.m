%%
%Require:mexopencv3.0
%��߶�ģ��ƥ��
%ԭ������ԭʼģ��ͼ��Ԥ�Ȳ���һϵ�в�ͬ�߶ȵ�ģ��
%����ʱ��ֱ������������߶ȵ�ģ�����ͼ��

%%
clear;
clc;

srcImg_rgb=imread('0132.jpg');
tImg=imread('hh.jpg');
srcImg=rgb2gray(srcImg_rgb);

[tImg_rows,tImg_cols]=size(tImg);
sprintf('ԭʼģ��ĳ߶�(�У���)Ϊ:[%d,%d]',tImg_rows,tImg_cols)

%�ܹ�10�ֳ߶�
scale_step=1.1;
%һϵ�в�ͬ�߶�
size_multi=zeros(5,2);
size_multi(1,:)=[tImg_rows,tImg_cols]; %ԭʼ�߶�
%smaller;
for i=1:5
    size_multi(i+1,:)=[tImg_rows,tImg_cols]/(scale_step^i);
end
%larger
for i=1:5
    size_multi(i+5,:)=[tImg_rows,tImg_cols]*(scale_step^i);
end

%�߶�תΪ����
size_multi=uint16(size_multi);

%��߶�ƥ��
resMap=cell(10,1);
peak=zeros(10,1);
position=cell(10,1);
figure('name','��߶�ģ��');
for i=1:10
    %ģ��߶ȱ任
    tempImg=imresize(tImg,size_multi(i,:));
    %��ʾģ��
    subplot(2,5,i);imshow(tempImg);
    %���ƥ��
    resMap{i}=cv.matchTemplate(srcImg,tempImg,'Method','CCorrNormed');
    %��λ��ֵ
    peak(i)=max(max(resMap{i}));
    %��ֵ��Ӧ���У���
    [row,col]=find(resMap{i}==peak(i));
    position{i}=[row,col];
end


%��ʾ�����߶ȵ�ƥ����
%figure('name','matchResult');
for i=1:10
    %���ƾ���
    showImg=cv.rectangle(srcImg_rgb,[position{i}(2),position{i}(1),size_multi(i,2),size_multi(i,1)],'Color',[255,255,0]);
    figure(i+1)
    imshow(showImg)
    sprintf('�߶�%d��ƥ���ֵΪ:%f',i,peak(i))
    %subplot(2,5,i);subimage(showImg);
end

%��λ����ֵ
maxPeak=max(peak);
maxIndex=find(peak==maxPeak);
sprintf('����ֵΪ:%d',maxPeak)
sprintf('����ֵ��Ӧ��ģ��߶�(��ѳ߶�)Ϊ:[%d,%d]',size_multi(maxIndex,1),size_multi(maxIndex,2))
%��ʾ���
resultImg=cv.rectangle(srcImg_rgb,[position{maxIndex}(2),position{maxIndex}(1),size_multi(maxIndex,2),size_multi(maxIndex,1)],'Color',[255,255,0]);
figure('name','���ƥ��');
imshow(resultImg);

