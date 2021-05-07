clear all
close all
clc
%% *************初始相位**************
N = 512;
G = 2;
phi0 = peaks(N)*G; %模拟初始相位
figure(1)
surf(phi0,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('Initial Phase','FontSize',14)
set(figure(1),'name','Initial Phase 3D','Numbertitle','off');
phi = angle(exp(j*phi0));         %包裹相位
figure(2);
imshow(phi,[]);
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);%title('Wrapped Phase','FontSize',14)
set(figure(2),'name','Wrapped Phase','Numbertitle','off');
axis on
%% *************相位解包裹**************
[m,n] = size(phi);
phidx=zeros(m,n);
phidy=zeros(m,n);
phidx(1:m-1,:)= angle(exp(j*(phi(2:m,:)-phi(1:m-1,:))));
phidy(:,1:n-1)= angle(exp(j*(phi(:,2:n)-phi(:,1:n-1))));
%********************对包裹相位求二阶偏微分**************
Rou3 = zeros(m,n);
Rou3dx = zeros(m,n);
Rou3dy = zeros(m,n);
Rou3dx(1:m-1,:) = phidx(2:m,:)-phidx(1:m-1,:);
Rou3dy(:,1:n-1) = phidy(:,2:n)-phidy(:,1:n-1);
Rou3 = Rou3dx + Rou3dy;
figure(3);
surf(Rou3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('lou3','FontSize',14)
set(figure(3),'name','R(x,y) 3D','Numbertitle','off');
figure(4);
imshow(Rou3,[]);
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);
set(figure(4),'name','R(x,y) 2D','Numbertitle','off');
%% ***********************DCT求解泊松方程********************
tic
PP3 = dct2(Rou3);
for ii=1:m
    for jj=1:n
        k1=2*cos((ii-1)*pi/(m));
        k2=2*cos((jj-1)*pi/(n));
        KK = k1+k2-4;
        PH3(ii,jj) = PP3(ii,jj)/KK;
    end
end
PH3(1,1) = -(PH3(1,2) + PH3(2,1) - PP3(1,1))/2;
phi3 = idct2(PH3);
toc
phi3 = phi3(1:m,1:n); %解包裹出的相位


figure(5);
surf(phi3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('BLS Phase Unwrapping','FontSize',14)
set(figure(5),'name','BLS Phase Unwrapping','Numbertitle','off');
————————————————
版权声明：本文为CSDN博主「James_Ray_Murphy」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/james_ray_murphy/article/details/79062226