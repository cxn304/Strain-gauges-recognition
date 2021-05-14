clear
close all
%% read files, 这个程序太棒了
for i = 3:3    % t3目前是有问题的
    imgDir=['./trainx_mat_t' num2str(i) '/'];    % wrapped 文件夹
    realDir = ['./trainy_mat_t' num2str(i) '/']; % unwrapped 文件夹
    saveDir = ['./ls_data/image_t' num2str(i) '_ls/'];
    usefolders = find_folders(imgDir);
    len = length(usefolders);
    realfolders = find_folders(realDir);
    for j = 1:len
        real_img_name = [realDir realfolders{j}];
        wrapped_name = [imgDir usefolders{j}];
        saved_name = [saveDir usefolders{j}];
        phi = load(wrapped_name);
        phase = phi.data;
        real_img = load(real_img_name);
        phase0 = real_img.data;
        %% generate original phase
        [x,y]=size(phase0);
        %% generate wrapped phase and mask
        phase_wrapped=angle(exp(1i*phase));
        MASK_in=ones(x,y);
        phase_in=phase_wrapped.*MASK_in;     %the wrapped phase to be unwrapped
%         figure(2);
%         imagesc(phase_in),axis image, axis off, colormap(gray);
%         disp('Select 1 phase_known point on the wrapped phase map');
        % [XYG]=ginput(1);
        if i==2||i==5
            XC=round(0);YC=round(0);    % t\t2\t3\t5用0,0,其他的用x/2 y/2,t2容易出现混叠
        else
            XC=round(x/2);YC=round(y/2);
        end
        %% unwrapping
        Nmax_ite=100;      %the max numbers of iterations
        Err=0.01;          %the threshold of the unwrapped phase error
        Calibration=false;% =false (if sigma<=0.8); =true (if sigma>0.8)
        [PU,PC,N_unwrap,t_unwrap]=CPULSI(phase_in,MASK_in,Nmax_ite,Err,XC,YC,Calibration); %unwrapping by MPULSI
        phase_unwrapped=PC.*MASK_in;
        save(saved_name,'phase_unwrapped');
        % phase_unwrapped=phase_unwrapped+mean(mean(phase(1:10,1:10)-phase_unwrapped(1:10,1:10)));
        error=phase_unwrapped-phase0;
        row128 = phase_unwrapped(128,:) - phase0(128,:);
        plot_imgs(phase,phase0,phase_unwrapped,row128,error,saved_name)
        close all
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
function plot_imgs(phi,real_img,phi3,row128,error_hole,saved_name)
h=figure;
set(gcf,'position',[0,50,900,550]);
subplot(2,3,1)
imshow(phi,[]);
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);%title('Wrapped Phase','FontSize',9)
title('Wrapped Phase','FontSize',9);

subplot(2,3,2)
surf(real_img,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);zlabel('Phase/Radians','FontSize',9);%title('BLS Phase Unwrapping','FontSize',9)
title('Real phase','FontSize',9);
axis on
subplot(2,3,3)
surf(phi3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);zlabel('Phase/Radians','FontSize',9);%title('BLS Phase Unwrapping','FontSize',9)
title('BLS Phase Unwrapping','FontSize',9);

subplot(2,3,4)
plot(row128)
title('Row 128 error','FontSize',9);

subplot(2,3,5)
surf(error_hole,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);zlabel('Phase/Radians','FontSize',9);
title('Full field error','FontSize',9);
axis normal;
saved_img = saved_name;
saved_img(end-2:end)='png';
saveas(h, saved_img, 'png');

end