clear all
close all
clc
%% read files
imgDir='./trainx_mat_t3/';    %总文件夹
usefolders = find_folders(imgDir);
len = length(usefolders);
wrapped_name = [imgDir usefolders{40}];
phi = load(wrapped_name);
phi = phi.data;
figure(2);
imshow(phi,[]);
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);%title('Wrapped Phase','FontSize',14)
set(figure(2),'name','Wrapped Phase','Numbertitle','off');
axis on
%%
[m,n] = size(phi);
zp = [m/2,n/2];
thing_mask = ones(m,n);
[deri,~]=derical(phi,thing_mask,zp);
unwrap=goodscan(deri,thing_mask,phi,1,m/2,n/2,m/2,n/2);
figure(5);
surf(unwrap,'FaceColor','interp', 'EdgeColor','none','FaceLighting','phong');
camlight left, axis tight
xlabel('X/Pixels','FontSize',14);ylabel('Y/Pixels','FontSize',14);zlabel('Phase/Radians','FontSize',14);%title('BLS Phase Unwrapping','FontSize',14)
set(figure(5),'name','QG Phase Unwrapping','Numbertitle','off');
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
function [A,Aorigin]=derical(phi,im_mask,zp)
%phi阶段相位，im_mask掩膜；zp十字中心
%质量图，【0,1】，1比较好；后面的1.5；0.9可调
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
Aorigin=A;%噪声大小没有上限，噪声越大A越大；
yuzhiqu=A(zp(1)-5:zp(1)+5,zp(2)-5:zp(2)+5);
yuzhiqu=sort(yuzhiqu(:));
yuzhi=yuzhiqu(fix(sum(~isnan(yuzhiqu))*0.9))*1.5;

A=yuzhi./max(A,yuzhi);
end
%%
function unwrap=goodscan(deri,mask,phi,i,clx,cly,crx,cry)
% maxline=20;
dimx=size(phi,1);   % row
dimy=size(phi,2);   % col
% linumber=dimx*3;
mask=(0./fix(deri)+1).*mask;
mask(mask==0)=nan;
% [r,rowref ]=max(sum(mask,2));
phi=mask.*phi;%phi; area of quality==1
C=~isnan(phi);%对非计算域进行膨胀操作，找出连通路径A4=imdilate(A3,se)
if i<=2
    cloumnref = clx;
else
    cloumnref = crx;
end
cc=C(:,cloumnref);
k=find(cc==1);
rowref=k(fix(numel(k)/2));  % fix Round towards zero.numel返回数组中元素个数
colref=cloumnref;   % 相当于列的零点
edl=cloumnref+2;
unwrap=nan(dimx,dimy);
unwrap(rowref,colref)=phi(rowref, colref);  % ?????
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
unwrap(:,cloumnref:edl) = GuidedFloodFill3(phi(:,cloumnref:edl), ...
    unwrap(:,cloumnref:edl), adjoin,[]);
adjoin=nan(dimx,1);
if edl==dimy-2
else
    for i=edl+1:dimy-2
        unwrap(:,i)=unwrap(:,i-1)+phi(:,i)-phi(:,i-1)-2*pi*round(...
            (phi(:,i)-phi(:,i-1))/2/pi);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))...
            ==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),1);%up
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==...
            -1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),-1);
    end
end
if cloumnref==3
else
    for i=cloumnref-1:-1:3
        unwrap(:,i)=unwrap(:,i+1)+phi(:,i)-phi(:,i+1)-2*pi*round((phi(:,i)-phi(:,i+1))/2/pi);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==1&~isnan(phi(3:dimx-2,i)))+2;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),1);
        l=find(isnan(unwrap(3:dimx-2,i))-isnan(unwrap(4:dimx-1,i))==-1&~isnan(phi(4:dimx-1,i)))+3;
        adjoin(1:length(l))=l;
        unwrap(:,i)=unwrapline(phi(:,i), unwrap(:,i), adjoin,length(l),-1);
    end
end
end
%%
function IM_unwrapped = GuidedFloodFill3(IM_phase, IM_unwrapped, ...
    adjoin ,derivative_variance)

if size(adjoin,1)==0
    IM_unwrapped =IM_unwrapped ;
else
    k=0;%adjoin指针%%!!!!!!!!!!!!!!!!!!!!!!!!!!!对棋盘格不适用，需要扩展四个角
    while ~isnan(adjoin(k+1,1))
        k=k+1;
    end
    %%derivative_varianc,IM_phase,IM_unwrapped are masked
    [r_dim, c_dim] = size(IM_phase);
    if ~isempty(derivative_variance)
        % Include edge pixels%未验证
        adjoinmap=ones([r_dim, c_dim]);
        adjoinmap(adjoin(1:k))=0;%若矩阵adjoin中为0，则已经在adjoin list中
        
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
            %First search below for an adjoining unwrapped phase pixel
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active+1, c_active))
                    
                    phase_ref(1) = IM_unwrapped(r_active+1, c_active)...
                        +IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference unwrapped phase
                    qualityneighbor(1)=derivative_variance(r_active+1, c_active);
                    
                else % unwrapped_binary(r_active+1, c_active)==0未展开
                    if(~isnan(IM_phase(r_active+1, c_active))*adjoinmap(r_active+1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active+1, c_active);  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        adjoinmap(r_active+1, c_active)=0;
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active-1, c_active))==1
                    
                    phase_ref(2) = IM_unwrapped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                    qualityneighbor(2)=derivative_variance(r_active-1, c_active);
                    %Obtain the reference unwrapped phase
                    %       D = IM_phase(r_active, c_active)-phase_ref;
                    %       deltap = atan2(sin(D),cos(D));   % Make it modulo +/-pi
                    %       phasev(2) = phase_ref + deltap;  % This is the unwrapped phase
                    %       IM_magv(2)= IM_mag(r_active-1, c_active);
                else % unwrapped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active))*adjoinmap(r_active-1, c_active)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active-1, c_active);
                        adjoinmap(r_active-1, c_active)=0;
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active+1))
                    
                    phase_ref(3) = IM_unwrapped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                    qualityneighbor(3)=derivative_variance(r_active, c_active+1);
                    
                else % unwrapped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1))*adjoinmap(r_active, c_active+1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active+1);
                        adjoinmap(r_active, c_active+1)=0;
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active-1))
                    
                    phase_ref(4) = IM_unwrapped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                    qualityneighbor(4)=derivative_variance(r_active, c_active-1);
                    
                else % unwrapped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1))*adjoinmap(r_active, c_active-1)==1)
                        k=k+1;
                        adjoin(k)=sub2ind(size(derivative_variance),r_active, c_active-1);
                        adjoinmap(r_active, c_active-1)=0;
                    end
                end
            end
            
            [IM_max,m] = max(qualityneighbor);
            %     idx_max = find((IM_magv >= 0.99*IM_max) & (idx_del==1));
            IM_unwrapped(r_active, c_active) =phase_ref(m);  % Use the first, if there is a tie
            
        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    else
        
        while k~= 0
            %input adjoin should be a pixel position i.e [3,3]
            %before run, add unwrapped start point(rowref, colref)and adjoin points
            %   if im_mask(rowref-1, colref, 1)==1;  adjoin=[rowref-1, colref;adjoin] end       %Mark the pixels adjoining the selected point
            % if im_mask(rowref+1, colref, 1)==1;  adjoin(rowref+1, colref, 1) = 1; end
            % if im_mask(rowref, colref-1, 1)==1;  adjoin(rowref, colref-1, 1) = 1; end
            % if im_mask(rowref, colref+1, 1)==1;  adjoin(rowref, colref+1, 1) = 1; end
            r_active=adjoin(k,1);
            c_active=adjoin(k,2);
            adjoin(k,:)=NaN;
            k=k-1;
            
            
            
            if(r_active+1<=r_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active+1, c_active))
                    phase_ref = IM_unwrapped(r_active+1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active+1, c_active))/2/pi);       % Obtain the reference unwrapped phase
                else % unwrapped_binary(r_active+1, c_active)==0未展开
                    if(~isnan(IM_phase(r_active+1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active+1, c_active];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                        
                    end
                end
            end
            %Then search above
            if(r_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active-1, c_active))
                    phase_ref= IM_unwrapped(r_active-1, c_active)+IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active-1, c_active))/2/pi);
                else % unwrapped_binary(r_active-1, c_active)==0
                    if(~isnan(IM_phase(r_active-1, c_active)))
                        k=k+1;
                        adjoin(k,:)=[r_active-1, c_active];
                    end
                end
            end
            %Then search on the right
            if(c_active+1<=c_dim)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active+1))
                    phase_ref = IM_unwrapped(r_active, c_active+1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active+1))/2/pi);
                else % unwrapped_binary(r_active, c_active+1)==0
                    if(~isnan(IM_phase(r_active, c_active+1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active+1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
            %Finally search on the left
            if(c_active-1>=1)  % Is this a valid index?
                if ~isnan(IM_unwrapped(r_active, c_active-1))
                    phase_ref= IM_unwrapped(r_active, c_active-1)+IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1)...
                        -2*pi*round((IM_phase(r_active, c_active)-IM_phase(r_active, c_active-1))/2/pi);
                else % unwrapped_binary(r_active, c_active-1)==0
                    if(~isnan(IM_phase(r_active, c_active-1)))
                        k=k+1;
                        adjoin(k,:)=[r_active, c_active-1];  % Put the elgible, still-wrapped neighbors of this pixels in the adjoin set
                    end
                end
            end
            
            IM_unwrapped(r_active, c_active) =phase_ref;
            
        end % while sum(sum(adjoin(2:r_dim-1,2:c_dim-1))) ~= 0  %Loop until there are no more adjoining pixels
    end
    return
end
end
%%
function IM_unwrapped =unwrapline(IM_phase,IM_unwrapped,adjoin,k,direction)
r_dim= length(IM_phase);
%%%%%%%%%adjoin还是各矩阵是否更改
if direction==-1%downwards
    while k~=0 %Loop until there are no more adjoining pixels
        r_active=adjoin(k);
        adjoin(k)=NaN;
        k=k-1;
        %Then search above
        if(r_active-1>=1)  % Is this a valid index?
            if ~isnan(IM_unwrapped(r_active-1))
                IM_unwrapped(r_active)= IM_unwrapped(r_active-1)...
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
            if ~isnan(IM_unwrapped(r_active+1))
                IM_unwrapped(r_active) = IM_unwrapped(r_active+1)...
                    +IM_phase(r_active)-IM_phase(r_active+1)...
                    -2*pi*round((IM_phase(r_active)-IM_phase(...
                    r_active+1))/2/pi);       % Obtain the reference unwrapped phase
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