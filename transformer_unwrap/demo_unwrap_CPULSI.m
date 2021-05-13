clear
close all
%% read files, 这个程序太棒了
imgDir='./trainx_mat_t2/';    %总文件夹
real_Dir = './trainy_mat_t2/';
usefolders = find_folders(imgDir);
len = length(usefolders);
realfolders = find_folders(real_Dir);
real_img_name = [real_Dir realfolders{10}];
wrapped_name = [imgDir usefolders{10}];
phi = load(wrapped_name);
phase = phi.data;
real_img = load(real_img_name);
phase0 = real_img.data;
%% generate original phase
% phase0=4*peaks(1024);
[x,y]=size(phase0);
sigma=0.7;%standard diviation of noise
% noise=sigma*randn(size(phase0));
% phase=phase0+noise;
% figure(1);
% imagesc(phase);axis image; axis off; colormap(jet);colorbar
%% generate wrapped phase and mask
phase_wrapped=angle(exp(1i*phase));
MASK_in=ones(x,y);
phase_in=phase_wrapped.*MASK_in;     %the wrapped phase to be unwrapped
figure(2);
imagesc(phase_in),axis image, axis off, colormap(gray);
disp('Select 1 phase_known point on the wrapped phase map');
% [XYG]=ginput(1);
XC=round(0);YC=round(0);    % t2用这个,其他的用x/2 y/2,t2容易出现混叠
%% unwrapping
Nmax_ite=100;      %the max numbers of iterations
Err=0.01;          %the threshold of the unwrapped phase error
Calibration=false;% =false (if sigma<=0.8); =true (if sigma>0.8)
[PU,PC,N_unwrap,t_unwrap]=CPULSI(phase_in,MASK_in,Nmax_ite,Err,XC,YC,Calibration); %unwrapping by MPULSI
phase_unwrapped=PC.*MASK_in;
% phase_unwrapped=phase_unwrapped+mean(mean(phase(1:10,1:10)-phase_unwrapped(1:10,1:10)));
error=phase_unwrapped-phase0;
row128 = phase_unwrapped(128,:) - phase0(128,:);
plot_imgs(phase,phase0,phase_unwrapped,row128,error)
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
function plot_imgs(phi,real_img,phi3,row128,error_hole)
figure(2);
imshow(phi,[]);
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);%title('Wrapped Phase','FontSize',14)
set(figure(2),'name','Wrapped Phase','Numbertitle','off');
axis on
figure(4);
surf(real_img,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('BLS Phase Unwrapping','FontSize',14)
set(figure(4),'name','Real image','Numbertitle','off');
axis on
figure(5);
surf(phi3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('BLS Phase Unwrapping','FontSize',14)
set(figure(5),'name','BLS Phase Unwrapping','Numbertitle','off');
axis on

figure(6);
plot(row128)
set(figure(6),'name','Row 128 error','Numbertitle','off');
axis on
figure(7);
surf(error_hole,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);
set(figure(7),'name','Full field error','Numbertitle','off');
axis on

end