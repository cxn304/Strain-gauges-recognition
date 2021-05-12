clear all
close all
clc
%% ****************
N = 512;
G = 2;
%% read files
imgDir='./trainx_mat/';    %总文件夹
usefolders = find_folders(imgDir);
len = length(usefolders);
wrapped_name = [imgDir usefolders{10}];
phi = load(wrapped_name);
phi = phi.data;
%%
[m,n] = size(phi);
phidx=zeros(m,n);
phidy=zeros(m,n);
phidx(1:m-1,:)= angle(exp(j*(phi(2:m,:)-phi(1:m-1,:))));
phidy(:,1:n-1)= angle(exp(j*(phi(:,2:n)-phi(:,1:n-1))));
%
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
%% *******************************
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
phi3 = phi3(1:m,1:n);


figure(5);
surf(phi3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('BLS Phase Unwrapping','FontSize',14)
set(figure(5),'name','BLS Phase Unwrapping','Numbertitle','off');


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
