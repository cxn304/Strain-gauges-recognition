function [IM,im_mag]=dftphase(aveimg,pixelstep)
%8����pixelstep=8
%16����pixelstep=16 ÿ������֮������ڣ�2pi�����
%
[row,colomn]=size(aveimg);
cycle=fix(pixelstep/4); % ��x��0����ȡ��
% ������λ
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