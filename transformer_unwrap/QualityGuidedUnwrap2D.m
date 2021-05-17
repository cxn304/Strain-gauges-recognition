clear
close all

%% REPLACE WITH YOUR IMAGES
for i = 6:6    % t3目前是有问题的
    imgDir=['./trainx_mat_t' num2str(i) '/'];    % wrapped 文件夹
    realDir = ['./trainy_mat_t' num2str(i) '/']; % unwrapped 文件夹
    saveDir = ['./qg_data/image_t' num2str(i) '_qg/'];
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
        im_mask=ones(size(phase));
        phase_unwrapped=zeros(size(phase));               %Zero starting matrix for unwrapped phase
        adjoin=zeros(size(phase));                     %Zero starting matrix for adjoin matrix
        unwrapped_binary=zeros(size(phase));           %Binary image to mark unwrapped pixels
        %% Calculate phase quality map
        phase_quality=PhaseDerivativeVariance(phase);
        
        %% Identify starting seed point on a phase quality map
        minp=phase_quality(2:end-1, 2:end-1); minp=min(minp(:));
        maxp=phase_quality(2:end-1, 2:end-1); maxp=max(maxp(:));
        [xpoint,ypoint] = size(phase);
        xpoint = 128;
        ypoint = 128;
        %% Unwrap
        colref=round(xpoint); rowref=round(ypoint);
        phase_unwrapped(rowref,colref)=phase(rowref,colref);                        %Save the unwrapped values
        unwrapped_binary(rowref,colref,1)=1;
        if im_mask(rowref-1, colref, 1)==1 adjoin(rowref-1, colref, 1)=1; end       %Mark the pixels adjoining the selected point
        if im_mask(rowref+1, colref, 1)==1 adjoin(rowref+1, colref, 1)=1; end
        if im_mask(rowref, colref-1, 1)==1 adjoin(rowref, colref-1, 1)=1; end
        if im_mask(rowref, colref+1, 1)==1 adjoin(rowref, colref+1, 1)=1; end
        phase_unwrapped=GuidedFloodFill(phase, phase_unwrapped, unwrapped_binary, phase_quality, adjoin, im_mask);    %Unwrap
        save(saved_name,'phase_unwrapped');
        error=phase_unwrapped-phase0;
        row128 = phase_unwrapped(128,:) - phase0(128,:);
        plot_imgs(phase,phase0,phase_unwrapped,row128,error,saved_name)
        close all
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
title('QG Phase Unwrapping','FontSize',9);

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
function phase_unwrapped=GuidedFloodFill(phase, phase_unwrapped, unwrapped_binary, derivative_variance, adjoin, IM_mask)

[r_dim, c_dim]=size(phase);
isolated_adjoining_pixel_flag=1;                            %Only remains set if an isolated adjoining pixel exists

while sum(sum(adjoin(2:r_dim-1,2:c_dim-1)))~=0              %Loop until there are no more adjoining pixels
    adjoining_derivative_variance=derivative_variance.*adjoin + 100.*~adjoin;      %Derivative variance values of the adjoining pixels (pad the zero adjoining values with 100)
    [r_adjoin, c_adjoin]=find(adjoining_derivative_variance==min(min(adjoining_derivative_variance)));      %Obtain coordinates of the maximum adjoining unwrapped phase pixel
    r_active=r_adjoin(1);
    c_active=c_adjoin(1);
    isolated_adjoining_pixel_flag=1;                                                %This gets cleared as soon as the pixel is unwrapped
    if (r_active==r_dim || r_active==1 || c_active==c_dim || c_active==1)            %Ignore pixels near the border
        phase_unwrapped(r_active, c_active)=0;
        adjoin(r_active, c_active)=0;
    else
        %First search below for an adjoining unwrapped phase pixel
        if unwrapped_binary(r_active+1, c_active)==1
            phase_ref=phase_unwrapped(r_active+1, c_active);                                   %Obtain the reference unwrapped phase
            p=unwrap([phase_ref phase(r_active, c_active)]);                             %Unwrap the active pixel
            phase_unwrapped(r_active, c_active)=p(2);
            unwrapped_binary(r_active, c_active)=1;                                         %Mark the pixel as unwrapped
            adjoin(r_active, c_active)=0;                                                   %Remove it from the list of adjoining pixels
            isolated_adjoining_pixel_flag=0;
            %Update the new adjoining pixels:
            if r_active-1>=1 && unwrapped_binary(r_active-1, c_active)==0 && IM_mask(r_active-1, c_active)==1
                adjoin(r_active-1, c_active)=1;
            end
            if c_active-1>=1 && unwrapped_binary(r_active, c_active-1)==0 && IM_mask(r_active, c_active-1)==1
                adjoin(r_active, c_active-1)=1;
            end
            if c_active+1<=c_dim && unwrapped_binary(r_active, c_active+1)==0 && IM_mask(r_active, c_active+1)==1
                adjoin(r_active, c_active+1)=1;
            end
        end
        %Then search above
        if unwrapped_binary(r_active-1, c_active)==1
            phase_ref=phase_unwrapped(r_active-1, c_active);                                       %Obtain the reference unwrapped phase
            p=unwrap([phase_ref phase(r_active, c_active)]);                             %Unwrap the active pixel
            phase_unwrapped(r_active, c_active)=p(2);
            unwrapped_binary(r_active, c_active)=1;                                         %Mark the pixel as unwrapped
            adjoin(r_active, c_active)=0;                                                   %Remove it from the list of adjoining pixels
            isolated_adjoining_pixel_flag=0;
            %Update the new adjoining pixels:
            if r_active+1<=r_dim && unwrapped_binary(r_active+1, c_active)==0 && IM_mask(r_active+1, c_active)==1
                adjoin(r_active+1, c_active)=1;
            end
            if c_active-1>=1 && unwrapped_binary(r_active, c_active-1)==0 && IM_mask(r_active, c_active-1)==1
                adjoin(r_active, c_active-1)=1;
            end
            if c_active+1<=c_dim && unwrapped_binary(r_active, c_active+1)==0 && IM_mask(r_active, c_active+1)==1
                adjoin(r_active, c_active+1)=1;
            end
        end
        %Then search on the right
        if unwrapped_binary(r_active, c_active+1)==1
            phase_ref=phase_unwrapped(r_active, c_active+1);                                       %Obtain the reference unwrapped phase
            p=unwrap([phase_ref phase(r_active, c_active)]);                             %Unwrap the active pixel
            phase_unwrapped(r_active, c_active)=p(2);
            unwrapped_binary(r_active, c_active)=1;                                         %Mark the pixel as unwrapped
            adjoin(r_active, c_active)=0;                                                   %Remove it from the list of adjoining pixels
            isolated_adjoining_pixel_flag=0;
            %Update the new adjoining pixels:
            if r_active+1<=r_dim && unwrapped_binary(r_active+1, c_active)==0 && IM_mask(r_active+1, c_active)==1
                adjoin(r_active+1, c_active)=1;
            end
            if c_active-1>=1 && unwrapped_binary(r_active, c_active-1)==0 && IM_mask(r_active, c_active-1)==1
                adjoin(r_active, c_active-1)=1;
            end
            if r_active-1>=1 && unwrapped_binary(r_active-1, c_active)==0 && IM_mask(r_active-1, c_active)==1
                adjoin(r_active-1, c_active)=1;
            end
        end
        %Finally search on the left
        if unwrapped_binary(r_active, c_active-1)==1
            phase_ref=phase_unwrapped(r_active, c_active-1);                                       %Obtain the reference unwrapped phase
            p=unwrap([phase_ref phase(r_active, c_active)]);                             %Unwrap the active pixel
            phase_unwrapped(r_active, c_active)=p(2);
            unwrapped_binary(r_active, c_active)=1;                                         %Mark the pixel as unwrapped
            adjoin(r_active, c_active)=0;                                                   %Remove it from the list of adjoining pixels
            isolated_adjoining_pixel_flag=0;
            %Update the new adjoining pixels:
            if r_active+1<=r_dim && unwrapped_binary(r_active+1, c_active)==0 && IM_mask(r_active+1, c_active)==1
                adjoin(r_active+1, c_active)=1;
            end
            if c_active+1<=c_dim && unwrapped_binary(r_active, c_active+1)==0 && IM_mask(r_active, c_active+1)==1
                adjoin(r_active, c_active+1)=1;
            end
            if r_active-1>=1 && unwrapped_binary(r_active-1, c_active)==0 && IM_mask(r_active-1, c_active)==1
                adjoin(r_active-1, c_active)=1;
            end
        end
        if isolated_adjoining_pixel_flag==1
            adjoin(r_active,c_active)=0;                                                    %Remove the current active pixel from the adjoin list
        end
    end
end
end
%%
function derivative_variance=PhaseDerivativeVariance(phase, varargin)

[r_dim,c_dim]=size(phase);
if nargin>=2                                    %Has a mask been included? If so crop the image to the mask borders to save computational time
    IM_mask=varargin{1};
    [maskrows,maskcols]=find(IM_mask);          %Identify coordinates of the mask
    minrow=min(maskrows)-1;                     %Identify the limits of the mask
    maxrow=max(maskrows)+1;
    mincol=min(maskcols)-1;
    maxcol=max(maskcols)+1;
    width=maxcol-mincol;                        %Now ensure that the cropped area is square
    height=maxrow-minrow;
    if height>width
        maxcol=maxcol + floor((height-width)/2) + mod(height-width,2);
        mincol=mincol - floor((height-width)/2);
    elseif width>height
        maxrow=maxrow + floor((width-height)/2) + mod(width-height,2);
        minrow=minrow - floor((width-height)/2);
    end
    if minrow<1 minrow=1; end
    if maxrow>r_dim maxrow=r_dim; end
    if mincol<1 mincol=1; end
    if maxcol>c_dim maxcol=c_dim; end
    phase=phase(minrow:maxrow, mincol:maxcol);        %Crop the original image to save computation time
end

[dimx, dimy]=size(phase);
dx=zeros(dimx,dimy);
p=unwrap([phase(:,1) phase(:,2)],[],2);
dx(:,1)=(p(:,2) - phase(:,1))./2;                    %Take the partial derivative of the unwrapped phase in the x-direction for the first column
p=unwrap([phase(:,dimy-1) phase(:,dimy)],[],2);
dx(:,dimy)=(p(:,2) - phase(:,dimy-1))./2;            %Take the partial derivative of the unwrapped phase in the x-direction for the last column
for i=2:dimy-1
    p=unwrap([phase(:,i-1) phase(:,i+1)],[],2);
    dx(:,i)=(p(:,2) - phase(:,i-1))./3;              %Take partial derivative of the unwrapped phase in the x-direction for the remaining columns
end

dy=zeros(dimx,dimy);
q=unwrap([phase(1,:)' phase(2,:)'],[],2);
dy(1,:)=(q(:,2)' - phase(1,:))./2;                   %Take the partial derivative of the unwrapped phase in the y-direction for the first row
p=unwrap([phase(dimx-1,:)' phase(dimx,:)'],[],2);
dy(dimx,:)=(q(:,2)' - phase(dimx-1,:))./2;           %Take the partial derivative of the unwrapped phase in the y-direction for the last row
for i=2:dimx-1
    q=unwrap([phase(i-1,:)' phase(i+1,:)'],[],2);
    dy(i,:)=(q(:,2)' - phase(i-1,:))./3;             %Take the partial derivative of the unwrapped phase in the y-direction for the remaining rows
end

dx_centre=dx(2:dimx-1, 2:dimy-1);
dx_left=dx(2:dimx-1,1:dimy-2);
dx_right=dx(2:dimx-1,3:dimy);
dx_above=dx(1:dimx-2,2:dimy-1);
dx_below=dx(3:dimx,2:dimy-1);
mean_dx=(dx_centre+dx_left+dx_right+dx_above+dx_below)./5;

dy_centre=dy(2:dimx-1, 2:dimy-1);
dy_left=dy(2:dimx-1,1:dimy-2);
dy_right=dy(2:dimx-1,3:dimy);
dy_above=dy(1:dimx-2,2:dimy-1);
dy_below=dy(3:dimx,2:dimy-1);
mean_dy=(dy_centre+dy_left+dy_right+dy_above+dy_below)./5;

stdvarx=sqrt( (dx_left - mean_dx).^2 + (dx_right - mean_dx).^2 + ...
    (dx_above - mean_dx).^2 + (dx_below - mean_dx).^2 + (dx_centre - mean_dx).^2 );
stdvary=sqrt( (dy_left - mean_dy).^2 + (dy_right - mean_dy).^2 + ...
    (dy_above - mean_dy).^2 + (dy_below - mean_dy).^2 + (dy_centre - mean_dy).^2 );
derivative_variance=100*ones(dimx, dimy);                         %Ensure that the border pixels have high derivative variance values
derivative_variance(2:dimx-1, 2:dimy-1)=stdvarx + stdvary;

if nargin>=2                                                      %Does the image have to be padded back to the original size?
    [orig_rows, orig_cols]=size(IM_mask);
    temp=100*ones(orig_rows, orig_cols);
    temp(minrow:maxrow, mincol:maxcol)=derivative_variance;       %Pad the remaining pixels with poor phase quality values
    derivative_variance=temp;
end
end