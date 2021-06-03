clear
close all     % 这个是三维重建最后一步
path='./calculate_dis/david/';%相位图地址
load([path 'stereoParams'])
%%
IntrinsicMatrix_left = stereoParams.CameraParameters1.IntrinsicMatrix';
IntrinsicMatrix_right = stereoParams.CameraParameters2.IntrinsicMatrix';
TranslationOfCamera2 = stereoParams.TranslationOfCamera2;
RotationOfCamera2 = stereoParams.RotationOfCamera2;
I2 = [RotationOfCamera2 TranslationOfCamera2'];
H2=IntrinsicMatrix_right*I2;  %得到转换矩阵(r1,r2,r3,T)
H1=[IntrinsicMatrix_left,zeros(3,1)];
kp=inv(IntrinsicMatrix_left);   % 左相机内参求逆
%%
xyzmapx=nan(1024,1024);
xyzmapy=nan(1024,1024);
xyzmapz=nan(1024,1024);
vumatch=load([path 'vumatch.asc'],'-ascii');
v1=vumatch(:,1);
u1=vumatch(:,2);
v0=vumatch(:,3);
u0=vumatch(:,4);
xyz=nan(length(u1),3);
for i=1:length(u1)
    A=[H1(3,1)*v1(i)-H1(1,1) H1(3,2)*v1(i)-H1(1,2) H1(3,3)*v1(i)-H1(1,3);
        H1(3,1)*u1(i)-H1(2,1) H1(3,2)*u1(i)-H1(2,2) H1(3,3)*u1(i)-H1(2,3);
        H2(3,1)*v0(i)-H2(1,1) H2(3,2)*v0(i)-H2(1,2) H2(3,3)*v0(i)-H2(1,3)];
    b=[H1(1,4)-H1(3,4)*v1(i);H1(2,4)-H1(3,4)*u1(i);H2(1,4)-H2(3,4)*v0(i)];
    xyz(i,:)=A\b;
    xyzmapx(u1(i),v1(i))=xyz(i,1);
    xyzmapy(u1(i),v1(i))=xyz(i,2);
    xyzmapz(u1(i),v1(i))=xyz(i,3);
end
xyz1=xyz';
% save([calibpath 'xyzmapx.mat'],'xyzmapx');
% save([calibpath 'xyzmapy.mat'],'xyzmapy');
% save([calibpath 'xyzmapz.mat'],'xyzmapz');
% save([calibpath 'xyzyes.asc'],'xyz','-ascii');
xyzmapz = fillmissing(xyzmapz,'linear',2,'EndValues','nearest');
xyzmapx = fillmissing(xyzmapx,'linear',2,'EndValues','nearest');
xyzmapy = fillmissing(xyzmapy,'linear',2,'EndValues','nearest');

a=nanmean(xyzmapz,'all');   % 异常值处理
c=nanstd(xyzmapz,0,'all');
xyzmapz(xyzmapz<a-3*c|xyzmapz>a+3*c)=nan;
s=surfl(xyzmapx,xyzmapy,xyzmapz);
s.EdgeColor='None';
