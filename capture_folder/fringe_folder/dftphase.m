function [IM,im_mag]=dftphase(aveimg,pixelstep)
%8像素pixelstep=8
%16像素pixelstep=16 每四相移之间的周期（2pi）间隔
%
[row,colomn]=size(aveimg);
cycle=fix(pixelstep/4); % 让x向0靠近取整
% 计算相位
cossum=0;
msinsum=0;
im_mag=(aveimg(:,:,4)-aveimg(:,:,2)).^2+(aveimg(:,:,1)-aveimg(:,:,3)).^2;
for k=1:cycle
    bas=[0,pi/2,pi,1.5*pi]+2*pi/pixelstep*(k-1);
    csbas=cos(bas);
    msbas=-sin(bas);
    cossum=cossum+aveimg(:,:,4*k)*csbas(4)+aveimg(:,:,4*k-2)*csbas(2)+aveimg(:,:,4*k-3)*csbas(1)+aveimg(:,:,4*k-1)*csbas(3);
    msinsum=msinsum+aveimg(:,:,4*k)*msbas(4)+aveimg(:,:,4*k-2)*msbas(2)+aveimg(:,:,4*k-3)*msbas(1)+aveimg(:,:,4*k-1)*msbas(3);
end
IM=atan2(msinsum,cossum);
end