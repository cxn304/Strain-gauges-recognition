clear
close all
%% read files
for i = 3:3
    imgDir=['./trainx_mat_t' num2str(i) '/'];    % wrapped �ļ���
    realDir = ['./trainy_mat_t' num2str(i) '/']; % phase_unwrappedped �ļ���
    saveDir = ['./qg_data/image_t' num2str(i) '_qg/'];
    usefolders = find_folders(imgDir);
    len = length(usefolders);
    realfolders = find_folders(realDir);
    for j = 1:len
        real_img_name = [realDir realfolders{j}];
        wrapped_name = [imgDir usefolders{j}];
        saved_name = [saveDir usefolders{j}];
        phi = load(wrapped_name);
        phi = phi.data;
        real_img = load(real_img_name);
        real_img = real_img.data;
        [m,n] = size(phi);
        zp = [m/2,n/2];
        thing_mask = ones(m,n);
        [deri,~]=derical(phi,thing_mask,zp);
        phase_unwrapped=goodscan(deri,thing_mask,phi,1,m/2,n/2,m/2,n/2);
        save(saved_name,'phase_unwrapped');
        row128 = phase_unwrapped(128,:) - real_img(128,:);
        error_hole = phase_unwrapped-real_img;
        plot_imgs(phi,real_img,phase_unwrapped,row128,error_hole,saved_name);
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
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);zlabel('Phase/Radians','FontSize',9);%title('BLS Phase phase_unwrappedping','FontSize',9)
title('Real phase','FontSize',9);
axis on
subplot(2,3,3)
surf(phi3,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',9);ylabel('Y/Pixels','FontSize',9);zlabel('Phase/Radians','FontSize',9);%title('BLS Phase phase_unwrappedping','FontSize',9)
title('QG Phase phase unwrappedping','FontSize',9);

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
% ���folder
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
function [A,Aorigin]=derical(phi,im_mask,zp)
%phi�׶���λ��im_mask��Ĥ��zpʮ������
%����ͼ����0,1����1�ȽϺã������1.5��0.9�ɵ�
row=size(phi,1);   % row
col=size(phi,2);   % col
A=zeros(row,col);

du=phi(3:size(phi,1),2:size(phi,2)-1)-phi(2:size(phi,1)-1,2:size(phi,2)-1);
du=du-2*pi*round(du/2/pi);

mean_du = (du(2:row-3, 2:col-3)+du(2:row-3,1:col-4)+du(...
    2:row-3,3:col-2)+du(1:row-4,2:col-3)+du(3:row-2,2:col-3))./5;
% stdvaru = sqrt( (du(3:row-2, 3:col-2) - mean_du).^2 + (du(3:row-2,2:col-3) - mean_du).^2 + ...
%               (du(3:row-2,4:col-1) - mean_du).^2 + (du(2:row-3,3:col-2) - mean_du).^2 + (du(4:row-1,3:col-2) - mean_du).^2 );

dv=phi(2:size(phi,1)-1,3:size(phi,2))-phi(2:size(phi,1)-1,2:size(...
    phi,2)-1);
dv=dv-2*pi*round(dv/2/pi);

mean_dv = (du(2:row-3, 2:col-3)+ du(2:row-3,1:col-4)+ du(...
    2:row-3,3:col-2)+ du(1:row-4,2:col-3)+du(3:row-2,2:col-3))./5;
% stdvarv = sqrt( (dv(3:row-2, 3:col-2) - mean_dv).^2 + (dv(3:row-2,2:col-3) - mean_dv).^2 + ...
%               (dv(3:row-2,4:col-1) - mean_dv).^2 + (dv(2:row-3,3:col-2) - mean_dv).^2 + (dv(4:row-1,3:col-2) - mean_dv).^2 );

A(3:row-2, 3:col-2)=(sqrt( (du(2:row-3, 2:col-3) - mean_du).^2 + (...
    du(2:row-3,1:col-4) - mean_du).^2 + ...
    (du(2:row-3,3:col-2) - mean_du).^2 + (du(1:row-4,2:col-3) -...
    mean_du).^2 + (du(3:row-2,2:col-3) - mean_du).^2 )+...
    sqrt( (dv(2:row-3, 2:col-3) - mean_dv).^2 + (dv(2:row-3,1:col-4)...
    - mean_dv).^2 + ...
    (dv(2:row-3,3:col-2) - mean_dv).^2 + (dv(1:row-4,2:col-3)...
    - mean_dv).^2 + (dv(3:row-2,2:col-3) - mean_dv).^2 ))/4;

% A(3:row-2, 3:col-2)=0.25./max(A(3:row-2, 3:col-2),0.25);
Aorigin=A;%������Сû�����ޣ�����Խ��AԽ��
yuzhiqu=A(zp(1)-5:zp(1)+5,zp(2)-5:zp(2)+5);
yuzhiqu=sort(yuzhiqu(:));
yuzhi=yuzhiqu(fix(sum(~isnan(yuzhiqu))*0.9))*1.5;

A=yuzhi./max(A,yuzhi);
end
%%
function phase_unwrapped=goodscan(deri,mask,phi,i,clx,cly,crx,cry)
% maxline=20;
dimx=size(phi,1);   % row
dimy=size(phi,2);   % col
% linumber=dimx*3;
mask=(0./fix(deri)+1).*mask;
mask(mask==0)=nan;
% [r,rowref ]=max(sum(mask,2));
phi=mask.*phi;%phi; area of quality==1
C=~isnan(phi);%�ԷǼ�����������Ͳ������ҳ���ͨ·��A4=imdilate(A3,se)
if i<=2
    cloumnref = clx;
else
    cloumnref = crx;
end
cc=C(:,cloumnref);
k=find(cc==1);
rowref=k(fix(numel(k)/2));  % fix Round towards zero.numel����������Ԫ�ظ���
colref=cloumnref;   % �൱���е����
edl=cloumnref+2;
phase_unwrapped=nan(dimx,dimy);
phase_unwrapped(rowref,colref)=phi(rowref, colref);  % ?????
adjoin=nan(dimx*(edl-cloumnref+1),2);
k=0;
if ~isnan(phi(rowref, colref+1))==1&&colref+1<=edl
    k=k+1;adjoin(k,:)=[rowref, 2];
end
if ~isnan(phi(rowref-1, colref))==1
    k=k+1;adjoin(k,:)=[rowref-1, 1];
end       %Mark the pixels adjoining the selected point
if ~isnan(phi(rowref+1, colref))==1
    k=k+1;adjoin(k,:)=[rowref+1, 1];
end
phase_unwrapped(:,cloumnref:edl) = GuidedFloodFill3(phi(:,cloumnref:edl), ...
    phase_unwrapped(:,cloumnref:edl), adjoin,[]);
adjoin=nan(dimx,1);
if edl==dimy-2
else
    for i=edl+1:dimy-2
        phase_unwrapped(:,i)=phase_unwrapped(:,i-1)+phi(:,i)-phi(:,i-1)-2*pi*round(...
            (phi(:,i)-phi(:,i-1))/2/pi);
        l=find(isnan(phase_unwrapped(3:dimx-2,i))-isnan(phase_unwrapped(4:dimx-1,i))...
            ==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        phase_unwrapped(:,i)=phase_unwrappedline(phi(:,i), phase_unwrapped(:,i), adjoin,length(l),1);%up
        l=find(isnan(phase_unwrapped(3:dimx-2,i))-isnan(phase_unwrapped(4:dimx-1,i))==...
            -1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        phase_unwrapped(:,i)=phase_unwrappedline(phi(:,i), phase_unwrapped(:,i), adjoin,length(l),-1);
    end
end
if cloumnref==3
else
    for i=cloumnref-1:-1:3
        phase_unwrapped(:,i)=phase_unwrapped(:,i+1)+phi(:,i)-phi(:,i+1)-2*pi*round((phi(:,i)-phi(:,i+1))/2/pi);
        l=find(isnan(phase_unwrapped(3:dimx-2,i))-isnan(phase_unwrapped(4:dimx-1,i))==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        phase_unwrapped(:,i)=phase_unwrappedline(phi(:,i), phase_unwrapped(:,i), adjoin,length(l),1);
        l=find(isnan(phase_unwrapped(3:dimx-2,i))-isnan(phase_unwrapped(4:dimx-1,i))==-1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        phase_unwrapped(:,i)=phase_unwrappedline(phi(:,i), phase_unwrapped(:,i), adjoin,length(l),-1);
    end
end
end
%%
function IM_phase_unwrappedped = GuidedFloodFill3(IM_phase, IM_phase_unwrappedped, ...
    adjoin ,derivative_variance)

if size(adjoin,1)==0
    IM_phase_unwrappedped =IM_phase_unwrappedped ;
else
    k=0;%adjoinָ��%%!!!!!!!!!!!!!!!!!!!!!!!!!!!�����̸����ã���Ҫ��չ�ĸ���
    while ~isnan(adjoin(k+1,1))
        k=k+1;
    end
    %%derivative_varianc,IM_phase,IM_phase_unwrappedped are masked
    [r_dim, c_dim] = size(IM_phase);
    if ~isempty(derivative_variance)
        % Include edge pixels%δ��֤
        adjoinmap=ones([r_dim, c_dim]);
        adjoinmap(adjoin(1:k))=0;%������adjoin��Ϊ0�����Ѿ���adjoin list��
        
        while k~= 0
            %Loop until there are no more adjoining pixels %Derivative variance values of the adjoining pixels (pad the zero adjoining values with 100)
            [max_deriv_var,ma] = max(derivative_variance(adjoin(k:-1:1))); % the minimum derivative variance
            [r_active, c_active]=ind2sub(size(derivative_variance),adjoin(k+1-ma));
            adjoin(k+1-ma)=adjoin(k);
            adjoin(k)=NaN;
            k=k-1;
            
            phase_ref = nan(1,4);     % Initialize.  Will overwrite for valid pixels
            qualityneighbor= nan(1,4);
            %IM_magv  = nan(1,4);     % Initialize.  Will overwrite for valid pixels
            %First search below for an adjoining phase_unwrappedped phase pixel
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active+1, c_active))
                    
                    phase_ref(1) = IM_phase_unwrappedped(r_active+1, c_active)...
                        +IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference phase_unwrappedped phase
                    qualityneighbor(1)=derivative_variance(r_active+1, c_active);
                    
                else % phase_unwrappedped_binary(r_active+1, c_active)==0δչ��
                    if(~isnan(IM_phase(r_active+1, c_active))*adjoinmap(r_active+1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active+1, c_active);  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        adjoinmap(r_active+1, c_active)=0;
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active-1, c_active))==1
                    
                    phase_ref(2) = IM_phase_unwrappedped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                    qualityneighbor(2)=derivative_variance(r_active-1, c_active);
                    %Obtain the reference phase_unwrappedped phase
                    %       D = IM_phase(r_active, c_active)-phase_ref;
                    %       deltap = atan2(sin(D),cos(D));   % Make it modulo +/-pi
                    %       phasev(2) = phase_ref + deltap;  % This is the phase_unwrappedped phase
                    %       IM_magv(2)= IM_mag(r_active-1, c_active);
                else % phase_unwrappedped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active))*adjoinmap(r_active-1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active-1, c_active);
                        adjoinmap(r_active-1, c_active)=0;
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active, c_active+1))
                    
                    phase_ref(3) = IM_phase_unwrappedped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                    qualityneighbor(3)=derivative_variance(r_active, c_active+1);
                    
                else % phase_unwrappedped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1))*adjoinmap(r_active, c_active+1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active+1);
                        adjoinmap(r_active, c_active+1)=0;
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active, c_active-1))
                    
                    phase_ref(4) = IM_phase_unwrappedped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                    qualityneighbor(4)=derivative_variance(r_active, c_active-1);
                    
                else % phase_unwrappedped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1))*adjoinmap(r_active, c_active-1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active-1);
                        adjoinmap(r_active, c_active-1)=0;
                    end
                end
            end
            
            [IM_max,m] = max(qualityneighbor);
            %     idx_max = find((IM_magv >= 0.99*IM_max) & (idx_del==1));
            IM_phase_unwrappedped(r_active, c_active) =phase_ref(m);  % Use the first, if there is a tie
            
        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    else
        
        while k~= 0
            %input adjoin should be a pixel position i.e [3,3]
            %before run, add phase_unwrappedped start point(rowref, colref)and adjoin points
            %   if im_mask(rowref-1, colref, 1)==1;  adjoin=[rowref-1, colref;adjoin] end       %Mark the pixels adjoining the selected point
            % if im_mask(rowref+1, colref, 1)==1;  adjoin(rowref+1, colref, 1) = 1; end
            % if im_mask(rowref, colref-1, 1)==1;  adjoin(rowref, colref-1, 1) = 1; end
            % if im_mask(rowref, colref+1, 1)==1;  adjoin(rowref, colref+1, 1) = 1; end
            r_active=adjoin(k,1);
            c_active=adjoin(k,2);
            adjoin(k,:)=NaN;
            k=k-1;
            
            
            
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active+1, c_active))
                    phase_ref = IM_phase_unwrappedped(r_active+1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference phase_unwrappedped phase
                else % phase_unwrappedped_binary(r_active+1, c_active)==0δչ��
                    if(~isnan(IM_phase(r_active+1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active+1, c_active];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active-1, c_active))
                    phase_ref= IM_phase_unwrappedped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                else % phase_unwrappedped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active-1, c_active];
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active, c_active+1))
                    phase_ref = IM_phase_unwrappedped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                else % phase_unwrappedped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active+1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_phase_unwrappedped(r_active, c_active-1))
                    phase_ref= IM_phase_unwrappedped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                else % phase_unwrappedped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active-1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
            
            IM_phase_unwrappedped(r_active, c_active) =phase_ref;
            
        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    end
    return
end
end
%%
function IM_phase_unwrappedped =phase_unwrappedline(IM_phase,IM_phase_unwrappedped,adjoin,k,direction)
r_dim= length(IM_phase);
%%%%%%%%%adjoin���Ǹ������Ƿ����
if direction==-1%downwards
    while k~=0 %Loop until there are no more adjoining pixels
        r_active=adjoin(k);
        adjoin(k)=NaN;
        k=k-1;
        %Then search above
        if(r_active-1>=1)  % Is this a valid index?
            if ~isnan(IM_phase_unwrappedped(r_active-1))
                IM_phase_unwrappedped(r_active)= IM_phase_unwrappedped(r_active-1)...
                    +IM_phase(r_active)-IM_phase(r_active-1)...
                    -2*pi*round((IM_phase(r_active)-IM_phase(...
                    r_active-1))/2/pi);
                if (r_active+1<=r_dim)
                    if(~isnan(IM_phase(r_active+1)))
                        k=k+1;
                        adjoin(k)=r_active+1; % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
        end
        
    end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
elseif direction==1%'upwards'
    while k~=0 %Loop until there are no more adjoining pixels
        r_active=adjoin(k);
        adjoin(k)=NaN;
        k=k-1;
        
        if(r_active+1<=r_dim)  % Is this a valid index?
            if ~isnan(IM_phase_unwrappedped(r_active+1))
                IM_phase_unwrappedped(r_active) = IM_phase_unwrappedped(r_active+1)...
                    +IM_phase(r_active)-IM_phase(r_active+1)...
                    -2*pi*round((IM_phase(r_active)-IM_phase(...
                    r_active+1))/2/pi);       % Obtain the reference phase_unwrappedped phase
                if (r_active-1>0)
                    if(~isnan(IM_phase(r_active-1)))
                        k=k+1;
                        adjoin(k)=r_active-1;
                    end
                end
            end
        end
    end
end
return
end